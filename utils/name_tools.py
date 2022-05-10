import os


def join(former, latter):
    if former.endswith('-'):
        if latter[0] == '_':
            return former + latter[1:]
        else:
            return former + latter
    else:
        if latter[0] == '_':
            return former + latter
        else:
            return former + '_' + latter


def get_img_path(imgs_dir, img_title, suffix='.svg'):
    img_seg = img_title.replace(' ', '_')
    return os.path.join(imgs_dir, img_seg + suffix)
