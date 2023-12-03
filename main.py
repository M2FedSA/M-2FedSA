import os
import random
from conf import args_parser

from data.Dataset import init_dataset
import warnings
from torch.utils.tensorboard import SummaryWriter
from Main_Server import MainServer
from Fed_Server import FedServer
from Client import *


warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prRed(skk): print("\033[91m {}\033[00m".format(skk))

def prGreen(skk): print("\033[92m {}\033[00m".format(skk))

def prYellow(skk): print("\033[93m {}\033[00m".format(skk))

def prLightPurple(skk): print("\033[94m {}\033[00m".format(skk))

def prPurple(skk): print("\033[95m {}\033[00m".format(skk))




if __name__ == '__main__':

    args = args_parser()  # parameter setting

    set_seed(args.seed)

    writer = SummaryWriter('./logs/' + args.out_name)

    devices, lrs, wds = paramsofmodal(args)

    # define models on client and main server
    prYellow("Initialising the models......")
    clients_models = {}
    clients_models['visual'], clients_models['text'], _ = init_client_models(args)

    # prepare data
    prPurple("Initialising the Datasets......")
    train_Datasets = init_dataset(args.dataset, 'train', args)
    test_Datasets = init_dataset(args.dataset, 'test', args)

    # split client data
    dict_clients_train = {}
    dict_clients_test = {}
    dict_clients_train['visual'] = dataset_iid(train_Datasets['visual'], int(args.client_num_total / args.modal_num))
    dict_clients_test['visual'] = dataset_iid(test_Datasets['visual'], int(args.client_num_total / args.modal_num))

    dict_clients_train['text'] = dict_clients_train['visual']
    dict_clients_test['text'] = dict_clients_test['visual']

    # init users
    users = get_all_clients(args, dict_clients_train, dict_clients_test, clients_models, train_Datasets, test_Datasets, devices, lrs, wds)

    # init severs
    M_ser = MainServer(args, devices, lrs, wds)
    F_ser = FedServer(users)
    m_loss_queue = []
    for round in range(args.rounds):
        for user in users:
            client_v = user['visual']
            client_t = user['text']

            batch_acc_train_v = []
            batch_loss_train_v = []
            batch_acc_train_t = []
            batch_loss_train_t = []
            batch_count_train = 0

            mloss = []

            for epoch in range(args.epochs):
                len_batch = len(client_v.traindata)
                for batch_id, (data_v, data_t) in enumerate(zip(client_v.traindata, client_t.traindata)):
                    x_v, y_v = data_v
                    x_t, y_t = data_t
                    del data_v, data_t
                    client_A = {}
                    client_dB = {}
                    metrics = {}
                    uni_loss = {}
                    multi_loss = {}

                    client_A['visual'] = client_v.Forward1(x_v)
                    client_A['text'] = client_t.Forward1(x_t)


                    server_B, server_fuse, server_fuse_fea = M_ser.Forward2(client_A, devices)

                    client_dB['visual'], metrics['visual'], uni_loss['visual'], multi_loss['visual'] = client_v.Forward3andBackword1(server_B['visual'], server_fuse, server_fuse_fea, y_v)
                    client_dB['text'], metrics['text'], uni_loss['text'], multi_loss['text'] = client_t.Forward3andBackword1(server_B['text'], server_fuse, server_fuse_fea, y_t)

                    mloss.append((multi_loss['visual'].to(args.device)+multi_loss['text'].to(args.device))/2)

                    server_dA = M_ser.Backward2(client_dB, multi_loss, devices)

                    client_v.Backward3(server_dA['visual'])
                    client_t.Backward3(server_dA['text'])

                    batch_acc_train_v.append(metrics['visual'])
                    batch_acc_train_t.append(metrics['text'])
                    batch_loss_train_v.append(uni_loss['visual'])
                    batch_loss_train_t.append(uni_loss['text'])

                    batch_count_train += 1
                    if batch_count_train == len_batch:
                        acc_avg_train_v = sum(batch_acc_train_v) / len(
                            batch_acc_train_v)
                        loss_avg_train_v = sum(batch_loss_train_v) / len(batch_loss_train_v)
                        batch_acc_train_v = []
                        batch_loss_train_v = []

                        acc_avg_train_t = sum(batch_acc_train_t) / len(
                            batch_acc_train_t)
                        loss_avg_train_t = sum(batch_loss_train_t) / len(batch_loss_train_t)
                        batch_acc_train_t = []
                        batch_loss_train_t = []


                        batch_count_train = 0

                        prRed('Round{}-User{}-Client{} Train => \tAcc: {:.3f} \tLoss: {:.4f}'.format(round,client_v.u_id,
                                                                                                              client_v.c_id,
                                                                                                              acc_avg_train_v,
                                                                                                              loss_avg_train_v))
                        writer.add_scalar('Client' + str(client_v.c_id) + 'Loss/Train', loss_avg_train_v, round)
                        writer.add_scalar('Client' + str(client_v.c_id) + 'Acc/Train', acc_avg_train_v, round)

                        prPurple('Round{}-User{}-Client{} Train => \tAcc: {:.3f} \tLoss: {:.4f}'.format(round, client_t.u_id,
                                                                                                     client_t.c_id,
                                                                                                     acc_avg_train_t,
                                                                                                     loss_avg_train_t))
                        writer.add_scalar('Client' + str(client_t.c_id) + 'Loss/Train', loss_avg_train_t, round)
                        writer.add_scalar('Client' + str(client_t.c_id) + 'Acc/Train', acc_avg_train_t, round)

        if len(m_loss_queue) <= 10:
            m_loss_queue.append(sum(mloss)/len(mloss))
        else:
            print(m_loss_queue)
            m_loss_queue.pop(0)
            m_loss_queue.append(sum(mloss)/len(mloss))
            diffs = [abs(m_loss_queue[i] - m_loss_queue[i - 1]) for i in
                     range(len(m_loss_queue) - 1, len(m_loss_queue) - 11, -1)]
            mean_diff = torch.mean(torch.stack(diffs))
            std_diff = torch.std(torch.stack(diffs))
            coef_var = std_diff / mean_diff
            print(coef_var)
            if coef_var < 0.3:
                args.FUSE = True
                M_ser.fusion_model.requires_grad_(False)


        # Agg
        F_ser.agg_classifier(args, users)
        F_ser.agg_modal_ada(args, users)
        F_ser.agg_task_ada(args, M_ser)

        # test
        total_avg_acc = []
        total_avg_loss = []
        if round % args.test_round == 0:
            for u in users:
                for m in u.keys():
                    acc_avg_test, loss_avg_test = u[m].test(M_ser)
                    prGreen(
                        'Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(u[m].c_id, acc_avg_test,
                                                                                                 loss_avg_test))
                    total_avg_acc.append(acc_avg_test)
                    total_avg_loss.append(loss_avg_test)
            prPurple('Round{} Test => \tAcc: {:.3f} \tLoss: {:.4f}'.format(round, np.array(
                total_avg_acc).mean(), np.array(total_avg_loss).mean()))
            writer.add_scalar('Loss/Test', np.array(total_avg_loss).mean(), round)
            writer.add_scalar('Acc/Test', np.array(total_avg_acc).mean(), round)

    writer.close()

    # ===================================================================================

    prLightPurple("Training and Evaluation completed!")

    # ===============================================================================





