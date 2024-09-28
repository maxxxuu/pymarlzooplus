from envs import REGISTRY_availability as env_REGISTRY_availability


def import_error_pt_butterfly():
    raise ImportError("pettingzoo[butterfly] is not installed! "
                      "\nInstall it running: \npip install 'pettingzoo[butterfly]'==1.24.3")


def import_error_pt_atari():
    raise ImportError("pettingzoo[atari] is not installed! "
                      "\nInstall it running: \npip install 'pettingzoo[atari]'==1.24.3")


def check_env_installation(env_name, env_registry, logger):

    if env_name not in list(env_registry.keys()):
        if env_name in env_REGISTRY_availability:
            logger.console_logger.error(
                "\n###########################################"
                f"\nThe requirements for the selected type of environment '{env_name}' have not been installed! "
                "\nPlease follow the installation instruction in the README files."
                "\n###########################################"
            )
        else:
            logger.console_logger.error(
                "\n###########################################"
                f"\nThe selected type of environment '{env_name}' is not supported!"
                f"\nPlease choose one of the following: \n{env_REGISTRY_availability}"
                "\n###########################################"
            )
        exit(0)


