
from tools import *
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        x, y = self.dataset[self.idxs[item]]
        return x, y

class Client(object):

    def __init__(self, modal, encoder, classifier, args, dataset_train=None, dataset_test=None, idxs_train=None, idxs_test=None,
                 u_id=None, c_id=None, device=None, lr=None, wd=None):
        self.modal = modal
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
        self.lr = lr
        self.wd = wd
        self.args = args
        self.u_id = u_id
        self.c_id = c_id
        self.b_s = args.batch
        self.CE_loss = nn.CrossEntropyLoss()

        self.traindata = DataLoader(DatasetSplit(dataset_train, idxs_train), batch_size=self.b_s, shuffle=False)
        self.testdata = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=self.b_s, shuffle=False)

        self.batch_count_train = 0
        self.batch_count_test = 0
        self.loss_train_collect = []
        self.acc_train_collect = []

        self.optimizer_en = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optimizer_cla = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=self.wd)



    def Forward1(self, x):
        self.encoder.to(self.device)
        self.encoder.train()
        self.optimizer_en.zero_grad()

        if self.modal == 'visual':
            x = x.to(self.device)
            self.x_A = self.encoder(x)
        elif self.modal == 'text':
            x['input_ids'], x['attention_mask'] = x['input_ids'].to(self.device), x['attention_mask'].to(self.device)
            self.x_A = self.encoder(x['input_ids'], x['attention_mask'])


        client_x_A = self.x_A.clone().detach().requires_grad_(True)
        self.x_A.requires_grad_(True)
        return client_x_A

    def Forward3andBackword1(self, B, fuse, fuse_fea, y):
        self.classifier.to(self.device)
        self.classifier.train()
        self.optimizer_cla.zero_grad()

        unilogit = self.classifier(B.to(self.device))
        multilogit = fuse.to(self.device)
        y = y.to(self.device)

        uni_loss = self.CE_loss(unilogit, y.float())
        multi_loss = self.CE_loss(multilogit, y.float())

        if self.args.FUSE:
            KL_loss = F.kl_div(F.softmax(multilogit).log(), F.softmax(unilogit), reduction='sum')
            DT_loss = KL_loss / (uni_loss + multi_loss)
            FT_loss = cosine_similarity(torch.squeeze(B).to(self.device), fuse_fea.to(self.device))
            total_loss = self.args.lambda1 * (uni_loss + multi_loss / 2) + self.args.lambda2 * DT_loss + self.args.lambda3 * (1 - FT_loss.mean())
        else:
            total_loss = uni_loss

        total_loss.backward()
        dB = B.grad.clone().detach()
        self.optimizer_cla.step()

        metric = calculate_auc(unilogit, y)

        return dB, metric, total_loss, multi_loss

    def Backward3(self, dA):

        self.x_A.backward(dA)
        self.optimizer_en.step()

    def test(self, M_ser):
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.encoder.eval()
        self.classifier.eval()
        batch_acc_test = []
        batch_loss_test = []

        with torch.no_grad():
            len_batch = len(self.testdata)
            for batch_id, (x, y) in enumerate(self.testdata):
                if self.modal == 'text':
                    y = y.to(self.device)
                    x['input_ids'] = x['input_ids'].to(self.device)
                    x['attention_mask'] = x['attention_mask'].to(self.device)
                else:
                    x, y = x.to(self.device), y.to(self.device)

                if self.modal == 'visual':
                    if self.args.dataset == 'HM':
                        client_fx = self.encoder(x)
                    else:
                        x = torch.permute(x, (0, 4, 1, 2, 3))
                        client_fx = self.encoder(x)
                elif self.modal == 'text':
                    client_fx = self.encoder(x['input_ids'], x['attention_mask'])
                else:
                    client_fx = self.encoder(x)

                dfx = M_ser.test(client_fx, self.modal, self.device)

                logits = self.classifier(dfx)
                loss = self.CE_loss(logits, y.float())

                if self.args.dataset == 'EPIC':
                    acc = calculate_accuracy(logits, y)
                elif self.args.dataset == 'HM':
                    acc = calculate_auc(logits, y)
                else:
                    acc = compute_uar(logits, y)

                batch_loss_test.append(loss.item())

                if self.args.dataset == 'HM':
                    batch_acc_test.append(acc)
                else:
                    batch_acc_test.append(acc.item())

                self.batch_count_test += 1
                if self.batch_count_test == len_batch:
                    acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
                    loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)
                    batch_acc_test = []
                    batch_loss_test = []
                    self.batch_count_test = 0

            return acc_avg_test, loss_avg_test





def get_all_clients(args, dict_clients_train, dict_clients_test, clients_models, train_Datasets, test_Datasets, devices, lrs, wds):

    users_num = int(args.client_num_total / args.modal_num)
    users = []
    clients_in_user = {}
    modalities = dict_clients_train.keys()
    c_num = 0
    for u in range(users_num):
        for modal in modalities:
            clients_in_user[modal] = Client(modal=modal,
                                            encoder=clients_models[modal][0],
                                            classifier=clients_models[modal][1],
                                            args=args,
                                            dataset_train=train_Datasets[modal],
                                            dataset_test=test_Datasets[modal],
                                            u_id = u,
                                            c_id = c_num,
                                            idxs_train=dict_clients_train[modal][u],
                                            idxs_test=dict_clients_test[modal][u],
                                            device=devices[modal],
                                            lr=lrs[modal],
                                            wd=wds[modal])
            c_num = c_num + 1
        users.append(clients_in_user)
        clients_in_user = {}

    assert args.client_num_total == c_num, "number of client define error!!!!!!"

    return users



