from tools import *

class MainServer(object):
    def __init__(self, args, devices, lrs, wds):
        self.args = args
        self.models = {}
        self.models['visual'], self.models['text'], self.models['audio'] = init_server_models(args)

        self.optims = {}

        for m in self.models.keys():
            if self.models[m] != None:
                self.optims[m] = torch.optim.AdamW(self.models[m].parameters(), lr=lrs[m], weight_decay=wds[m])
        self.models = remove_none(self.models)

        self.fusion_model = MultiHeadA(args.heads, args.fusion_in_dim, args.fusion_hid_dim, args.class_num, args)
        self.optims_fuse = torch.optim.AdamW(self.fusion_model.parameters(), lr=self.args.lr, weight_decay=args.f_weight_decay)


    def Forward2(self, A, devices):
        self.client_A = A
        server_B = {}
        self.B = {}
        for m in self.models.keys():
            self.models[m].to(devices[m])
            self.models[m].train()
            self.optims[m].zero_grad()


            if m == 'visual':
                self.B['visual'] = self.models[m](self.client_A[m])
                server_B['visual'] = self.B['visual'].clone().detach().requires_grad_(True)
            elif m == 'text':
                self.B['text'] = self.models[m](self.client_A[m])
                self.B['text'] = self.B['text']['pooler_output']
                server_B['text'] = self.B['text'].clone().detach().requires_grad_(True)
            else:
                self.B['audio'] = self.models[m](self.client_A[m])
                server_B['audio'] = self.B['audio'].clone().detach().requires_grad_(True)

        # Fusion
        self.Fusion()

        server_fuse = self.fusion_logit.clone().detach().requires_grad_(True)
        server_fuse_fea = self.fusion_fea.clone().detach()
        server_fuse_fea = server_fuse_fea.requires_grad_(False)

        return server_B, server_fuse, server_fuse_fea

    def Backward2(self, dB, multi_loss, devices):
        server_dA = {}
        for m in self.models.keys():
            self.B[m].backward(dB[m])
            server_dA[m] = self.client_A[m].grad.clone().detach()
            self.optims[m].step()

        if self.args.FUSE == False:
            totalmloss = (multi_loss['visual'].to(self.args.device) + multi_loss['text'].to(self.args.device))/2
            totalmloss.backward()
            self.optims_fuse.step()

        return server_dA

    def Fusion(self):
        cat_list = []
        if self.B['visual'].size(0) != 1:
            for m in self.models.keys():
                if m != 'text':
                    cat_list.append(torch.squeeze(self.B[m]).to(self.args.device))
                else:
                    cat_list.append(self.B[m].to(self.args.device))
        else:
            for m in self.models.keys():
                if m != 'text':
                    cat_list.append(torch.unsqueeze(torch.squeeze(self.B[m]), dim=0).to(self.args.device))
                else:
                    cat_list.append(self.B[m].to(self.args.device))

        com_fea = torch.cat(cat_list, dim=1)
        self.fusion_model.to(self.args.device)
        self.optims_fuse.zero_grad()
        self.fusion_fea, self.fusion_logit = self.fusion_model(com_fea)

    def test(self, client_fx, modal, device):
        self.models[modal].to(device)
        self.models[modal].eval()
        with torch.no_grad():
            client_fx = client_fx.to(device)
            server_fx = self.models[modal](client_fx)
            if modal == 'text':
                server_fx = server_fx.pooler_output

        self.models[modal].to('cpu')

        return server_fx








