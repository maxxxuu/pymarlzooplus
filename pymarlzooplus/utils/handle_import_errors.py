import importlib.util

from pymarlzooplus.envs import REGISTRY_availability as env_REGISTRY_availability


def import_error_pt_butterfly():
    raise ImportError(
        "pettingzoo[butterfly] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[butterfly]'==1.24.3"
    )

def import_error_pt_atari():
    raise ImportError(
        "pettingzoo[atari] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[atari]'==1.24.3"
    )


def is_package_installed(package_name):
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

def atari_rom_error(e):
    # Check if the error message is about the ROM not being installed
    if "Please install roms using AutoROM tool" in str(e):
        if is_package_installed('AutoROM') is False:
            print(
                "The required Atari ROM is not installed. Please install the ROMs using the AutoROM tool."
                "\nYou can install AutoROM by running:"
                "\npip install autorom"
                "\nThen, to automatically download and install Atari ROMs, run:"
                "\nAutoROM -y"
            )
        else:
            raise OSError(
                "The required Atari package 'autorom' is installed, but the Atari ROMs have not been downloaded!."
                "\nRun the following command in your terminal: \nAutoROM -y"
            )
    else:
        raise e

def import_error_pt_classic():
    raise ImportError(
        "pettingzoo[classic] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[classic]'==1.24.3"
    )

def import_error_pt_mpe():
    raise ImportError(
        "pettingzoo[mpe] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[mpe]'==1.24.3"
    )

def import_error_pt_sisl():
    raise ImportError(
        "pettingzoo[sisl] is not installed! "
        "\nInstall it running: \npip install 'pettingzoo[sisl]'==1.24.3"
    )

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


