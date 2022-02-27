import argparse
import configparser
import os

import torch


def get_training_parser() -> None:
    """
    Base training parser for lightning training
    """
    params = get_base_parser()
    # se è stato specificato il device "cuda" o "cpu"
    if hasattr(params, "device"):
        if hasattr(params, "cuda_visible"):
            print("Cuda device:", params.cuda_visible)
        else:
            if torch.cuda.is_available():
                params.cuda_visible = 0
                params.number_of_gpu_available = torch.cuda.device_count()
            if params.device == "cuda":
                os.environ["CUDA_VISIBLE_DEVICES"] = str(params.cuda_visible)
        torch.cuda.set_device(params.cuda_visible)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params.device = device
    return params


def get_base_parser() -> None:
    """
    Basic parser
    """
    parser = argparse.ArgumentParser()
    # Sezione relativa agli argomenti da passare allo script di avvio
    parser.add_argument("config_file", help="Config file name", default="config_model.ini")

    parser.add_argument(
        "-r",
        help="On which machines the script is running. Remote or Jupiter",
        required=False,
        action="store_true",
        dest="remote",
    )
    args = parser.parse_args()

    if hasattr(args, "debugging"):
        import pdb

        pdb.set_trace()
    # sezione per l'estrazione degli argomenti dal file di configurazione
    parser_raw = configparser.RawConfigParser()
    parser_raw.read_file(open(args.config_file))
    # params = argparse.ArgumentParser()
    params = args
    for k in parser_raw.keys():
        if k == "DEFAULT":
            continue
        for kk, vv in dict(parser_raw[k]).items():
            try:
                params.__dict__[kk] = eval(vv)
            except:
                params.__dict__[kk] = vv
    return params


def get_raw_parser(configuration_file_path):
    """
    Raw parser for local configuration file
    @deprecated use Hydra config
    Args:
        configuration_file_path:

    Returns:

    """
    parser_raw = configparser.RawConfigParser()
    parser_raw.read_file(open(configuration_file_path))
    # params = argparse.ArgumentParser()
    params = configparser.ConfigParser()
    for k in parser_raw.keys():
        if k == "DEFAULT":
            continue
        for kk, vv in dict(parser_raw[k]).items():
            try:
                params.__dict__[kk] = eval(vv)
            except:
                params.__dict__[kk] = vv
    return params


def check_params_required(params, required_params_list) -> None:
    """
    Check for required params
    Args:
        params: dictionary of parameters to run
        required_params_list: list of required params

    Raises:
        BaseException

    """
    err = 0
    fail = "\033[91m"
    endc = "\033[0m"
    for re in required_params_list:
        if not hasattr(params, re):
            print(f"{fail}Missing value {re}{endc}")
            err += 1
    if err > 0:
        print(f"{fail}Values required: {required_params_list}{endc}")
        raise BaseException(f"Missing parameters.")
