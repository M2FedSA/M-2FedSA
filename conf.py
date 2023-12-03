import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_name', type=str, default="", help='epochs of local client per round')
    # basic parameters
    parser.add_argument('--epochs', type=int, default=1, help='epochs of local client per round')
    parser.add_argument('--rounds', type=int, default=300, help='rounds of training')
    parser.add_argument('--frac', type=float, default=1, help='client sampling rate')
    parser.add_argument('--client_num_total', type=int, default=50, help='number of total clients')
    parser.add_argument('--dataset', type=str, default='HM', choices=["MELD", "EPIC", "HM"], help='name of dataset')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--modal_num', type=int, default=2, help='number of modal')
    parser.add_argument('--if_bn', type=bool, default=False, help='two loss trade-off')
    parser.add_argument('--class_num', type=int, default=2, choices=[7, 97, 2], help='number of classes for visual modality')
    parser.add_argument('--iid', type=float, default=0.0, help='')
    parser.add_argument('--test_round', type=int, default=10, help='')
    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--device', type=int, default=0, help='gpu id')
    parser.add_argument('--FUSE', type=bool, default=False, help='gpu id')



    # fusion
    parser.add_argument('--fusion_flag', type=bool, default=True, help='')
    parser.add_argument('--heads', type=int, default=8, choices=[4, 8], help="")
    parser.add_argument('--fusion_in_dim', type=int, default=1536, choices=[2048, 1280, 1536], help='')
    parser.add_argument('--fusion_hid_dim', type=int, default=256, choices=[512, 256, 128, 64], help='')
    parser.add_argument('--f_weight_decay', type=float, default=0.05, help="")


    # loss
    parser.add_argument('--fea_ds_flag', type=bool, default=False, help='feature dist')
    parser.add_argument('--lambda1', type=float, default=0.6, help='two loss trade-off')
    parser.add_argument('--lambda2', type=float, default=0.2, help='two loss trade-off')
    parser.add_argument('--lambda3', type=float, default=0.2, help='two loss trade-off')


    # path
    parser.add_argument('--data_root_path', type=str, default='', help='root path of dataset')
    parser.add_argument('--checkpoint_save_path', type=str, default='', help='save path of checkpoint')

    # visual modality
    parser.add_argument('--v_lr', type=float, default=1e-4, help='learning rate of visual modality')
    parser.add_argument('--v_class_num', type=int, default=2, choices=[7, 97, 2], help='number of classes for visual modality')
    parser.add_argument('--v_model_name', type=str, default='swinT', help='model name for visual modality')
    parser.add_argument("--v_optimizer", default="adamw", choices=["sgd", "adamw"], type=str, help="Ways for visual optimization.")
    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument('--v_client_num', type=int, default=25, help='number of clients for visual')
    parser.add_argument('--v_fea_dim', type=int, default=768, help='Input feature dimensions of the classifier')
    parser.add_argument('--v_hid_dim', type=int, default=128, help='Hidden feature dimensions of the classifier')
    parser.add_argument('--v_device', type=int, default=7, help='gpu id')
    parser.add_argument('--bs_v', type=int, default=32, help='batch_size')
    parser.add_argument('--v_weight_decay', type=float, default=0.05, help="")


    # text modality
    parser.add_argument('--t_lr', type=float, default=1e-4, help='learning rate of text modality')
    parser.add_argument('--t_class_num', type=int, default=2, choices=[7, 97, 2], help='number of classes for text modality')
    parser.add_argument('--t_model_name', type=str, default='RoBERTa', help='model name for text modality')
    parser.add_argument("--t_optimizer", default="adamw", choices=["sgd", "adamw"], type=str, help="Ways for text optimization.")
    parser.add_argument('--t_client_num', type=int, default=25, help='number of clients for text')
    parser.add_argument('--t_fea_dim', type=int, default=768, help='Input feature dimensions of the classifier')
    parser.add_argument('--t_hid_dim', type=int, default=128, help='Hidden feature dimensions of the classifier')
    parser.add_argument('--t_device', type=int, default=3, help='gpu id')
    parser.add_argument('--bs_t', type=int, default=32, help='batch_size')
    parser.add_argument('--t_weight_decay', type=float, default=0.05, help="")





    args = parser.parse_args()
    return args