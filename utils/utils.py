
def resize(img, target_size=224):
    # Resize keeping aspect ratio so longer side = target_size
    w, h = img.size
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    img = img.resize((new_w, new_h))
    return img