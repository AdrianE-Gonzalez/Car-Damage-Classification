import os

def get_subdirectories(path):
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]

def get_base_dir():
    return os.path.join(os.getcwd(), 'data')

def get_train_dir():
    return os.path.join(get_base_dir(), 'training')

def get_val_dir():
    return os.path.join(get_base_dir(), 'validation')

def get_list():
    return get_subdirectories(get_train_dir())