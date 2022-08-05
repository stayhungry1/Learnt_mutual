import os

def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size

# out_filepath = './feature/35_down2_bit/000a1249af2bc5f0.vvc'
# out_filepath = './feature/35_bit/000a1249af2bc5f0.vvc'
out_filepath = '000a1249af2bc5f0_yuv.vvc'
# height = 2210
# width = 2432
# height = 4250
# width = 4864
height = 678
width = 1024
a = filesize(out_filepath) * 8.0
bpp = filesize(out_filepath) * 8.0 / (height * width)
print('bit: %d' %a)
print('pixel: %d' %(height * width))
print('bit: %7.5f' %bpp)
#./DecoderAppStatic -b ./feature/35_down2_bit/000a1249af2bc5f0.vvc -o ./feature/000a1249af2bc5f0_rec.yuv > 000a1249af2bc5f0_rec.txt
