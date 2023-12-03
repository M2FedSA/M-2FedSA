import torch
import copy


class FedServer(object):
    def __init__(self, users):
        self.modalities = users[0].keys()

    def agg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def agg_classifier(self, args, users):

        cla_set = {}
        for m in self.modalities:
            cla_set[m] = []

        for u in users:
            for m in self.modalities:
                cla_set[m].append(u[m].classifier.state_dict())
        cla_new = {}
        for m in self.modalities:
            cla_new[m] = self.agg(cla_set[m])

        for u in users:
            for m in self.modalities:
                u[m].classifier.load_state_dict(cla_new[m])


    def agg_modal_ada(self, args, users):

        ma_set = {}
        for m in self.modalities:
            ma_set[m] = []

        for u in users:
            for m in self.modalities:
                state_dict = u[m].encoder.state_dict()
                adapter_params = {}

                for name, param in state_dict.items():
                    if "Adapter" in name:
                        adapter_params[name] = param
                ma_set[m].append(adapter_params)

        ma_new = {}
        for m in self.modalities:
            ma_new[m] = self.agg(ma_set[m])

        for u in users:
            for m in self.modalities:
                u[m].encoder.load_state_dict(ma_new[m], strict=False)


    def agg_task_ada(self, args, M_ser):
        ta_set = {}

        for m in self.modalities:
            state_dict = M_ser.models[m].state_dict()
            adapter_params = {}
            for name, param in state_dict.items():
                if "Update_mlp" in name:
                    adapter_params[name] = param
            ta_set[m] = adapter_params

        w_avg = copy.deepcopy(ta_set['visual'])

        for k1, k2 in zip(w_avg.keys(), ta_set['text'].keys()):
            w_avg[k1] = w_avg[k1].to(args.device)
            w_avg[k1] += ta_set['text'][k2].to(args.device)

            w_avg[k1] = torch.div(w_avg[k1], len(ta_set.keys()))

        for k, k1, k2 in zip(w_avg.keys(), ta_set['visual'].keys(), ta_set['text'].keys()):
            ta_set['visual'][k1] = w_avg[k]
            ta_set['text'][k2] = w_avg[k]


        for m in self.modalities:
            M_ser.models[m].load_state_dict(ta_set[m], strict=False)





