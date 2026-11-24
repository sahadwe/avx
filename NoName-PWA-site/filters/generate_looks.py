#!/usr/bin/env python3
"""Generates the No Name colour looks (3D LUTs).
Each look is written as its own file  filters/looks/<id>.js  (a 33x33x33 LUT stored as a 1089x33 PNG strip,
blue slices side by side) and listed in filters/looks.js.  All looks are original grades built from
curves, lift/gamma/gain, channel mixing, split-toning and hue-band shifts.
Run:  python3 filters/generate_looks.py     (needs numpy + Pillow)
To add a look: add an entry to LOOKS below, run the script, done."""
import base64, io, os, json
import numpy as np
from PIL import Image

N = 33
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'looks')

def luma(c): return c[..., 0]*0.2126 + c[..., 1]*0.7152 + c[..., 2]*0.0722
def clip(c): return np.clip(c, 0, 1)

def sat(c, s):
    l = luma(c)[..., None]; return l + (c - l)*s
def vib(c, v):  # saturate muted colours more than saturated ones
    mx = c.max(-1, keepdims=True); mn = c.min(-1, keepdims=True)
    s = (mx - mn); return sat(c, 1 + v*(1 - s))
def contrast(c, k, pivot=0.5):  # smooth S-curve, k in -1..1
    c = clip(c)
    if k >= 0:
        s = 1/(1 + np.exp(-(c - pivot)*(4 + 10*k)))
        lo = 1/(1 + np.exp((pivot)*(4 + 10*k))); hi = 1/(1 + np.exp(-(1 - pivot)*(4 + 10*k)))
        return (s - lo)/(hi - lo)*k + c*(1 - k)
    return c*(1 + k) + pivot*(-k)*(1 - 0) * 0 + (0.5 + (c - 0.5)*0.6)*(-k)
def lgg(c, lift=0.0, gamma=1.0, gain=1.0):
    lift = np.array(lift, dtype=np.float64); gain = np.array(gain, dtype=np.float64); gamma = np.array(gamma, dtype=np.float64)
    c = c*(gain - lift) + lift
    return np.power(clip(c), 1/gamma)
def fade(c, black=0.0, white=1.0): return black + c*(white - black)
def tone(c, sh=(0, 0, 0), hi=(0, 0, 0), bal=0.5, strength=1.0):
    l = luma(clip(c))[..., None]
    ms = np.clip(1 - l/ max(bal, 1e-3), 0, 1)**1.5 * 1.0
    mh = np.clip((l - bal)/ max(1 - bal, 1e-3), 0, 1)**1.5 * 1.0
    return c + strength*(ms*np.array(sh) + mh*np.array(hi))
def mix(c, m):
    return np.einsum('...j,ij->...i', c, np.array(m, dtype=np.float64))
def rgb2hsv(c):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = c.max(-1); mn = c.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm]/d[rm]) % 6; h[gm] = (b - r)[gm]/d[gm] + 2; h[bm] = (r - g)[bm]/d[bm] + 4
    h = h/6.0
    s = np.where(mx > 1e-9, d/np.maximum(mx, 1e-9), 0)
    return h, s, mx
def hsv2rgb(h, s, v):
    h6 = (h % 1.0)*6; i = np.floor(h6).astype(int) % 6; f = h6 - np.floor(h6)
    p = v*(1 - s); q = v*(1 - s*f); t = v*(1 - s*(1 - f))
    r = np.choose(i, [v, q, p, p, t, v]); g = np.choose(i, [t, v, v, q, p, p]); b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)
def hue_shift(c, center_deg, width_deg, shift_deg, sat_mul=1.0, val_mul=1.0):
    """Move colours near `center_deg` by `shift_deg` (smooth falloff), optionally scale their saturation/value."""
    c = clip(c); h, s, v = rgb2hsv(c)
    d = np.abs(((h*360 - center_deg) + 180) % 360 - 180)
    w = np.clip(1 - d/width_deg, 0, 1); w = w*w*(3 - 2*w)*np.clip(s*3, 0, 1)
    h2 = h + w*shift_deg/360.0; s2 = np.clip(s*(1 + w*(sat_mul - 1)), 0, 1); v2 = v*(1 + w*(val_mul - 1))
    return hsv2rgb(h2, s2, np.clip(v2, 0, 1))
def soft_clip(c, knee=0.8):  # filmic highlight roll-off
    c = np.asarray(c)
    over = np.maximum(c - knee, 0); return np.where(c > knee, knee + (1 - knee)*np.tanh(over/(1 - knee)), c)
def mono(c, w=(0.2126, 0.7152, 0.0722)):
    l = c[..., 0]*w[0] + c[..., 1]*w[1] + c[..., 2]*w[2]; return np.stack([l, l, l], -1)
def tint_mono(c, dark, light, w=(0.2126, 0.7152, 0.0722)):
    l = np.clip(c[..., 0]*w[0] + c[..., 1]*w[1] + c[..., 2]*w[2], 0, 1)[..., None]
    return np.array(dark)*(1 - l) + np.array(light)*l

# ---------------------------------------------------------------- the looks
def L_blockbuster(c):
    c = tone(c, sh=(-.04, .03, .10), hi=(.10, .04, -.08), strength=1.0)
    c = hue_shift(c, 25, 40, -4, 1.25); c = hue_shift(c, 200, 50, 8, 1.1)
    return soft_clip(contrast(vib(c, .25), .35), .85)
def L_tealorange(c):
    c = tone(c, sh=(-.09, .03, .12), hi=(.16, .06, -.13), strength=1.2); c = sat(c, 1.15)
    return soft_clip(contrast(c, .3), .86)
def L_moodyteal(c):
    c = mix(c, [[.92, .06, .02], [.02, .96, .02], [.0, .12, .88]]); c = tone(c, sh=(-.08, .04, .11), hi=(.04, .05, .02))
    return contrast(lgg(sat(c, .8), lift=.02, gamma=.92), .3)
def L_goldenhour(c):
    c = mix(c, [[1.08, .04, -.02], [.02, 1.0, -.02], [-.04, -.02, .86]]); c = tone(c, sh=(.03, .0, -.03), hi=(.08, .05, -.04))
    return soft_clip(contrast(vib(c, .3), .22), .88)
def L_bleach(c):
    l = luma(clip(c))[..., None]; d = sat(c, .45); c = d*0.55 + (c*l*1.0 + (1 - l)*c*0)*0.45 + 0
    return clip(contrast(c*1.04, .55))
def L_neonoir(c):
    c = tint_mono(c, (0.02, 0.03, 0.06), (0.88, 0.92, 1.0)); return contrast(c, .5)
def L_portra(c):
    c = mix(c, [[1.04, .02, -.02], [.0, 1.0, .0], [-.02, .0, .94]]); c = fade(c, .035, .97)
    c = tone(c, sh=(.0, .01, .02), hi=(.04, .02, -.02)); return soft_clip(contrast(vib(c, .1), .15), .9)
def L_fujigreen(c):
    c = hue_shift(c, 110, 50, -10, 1.1); c = tone(c, sh=(-.02, .03, .02), hi=(.0, .03, .0))
    return soft_clip(contrast(fade(c, .02, .98), .2), .9)
def L_70s(c):
    c = mix(c, [[1.0, .08, .0], [.0, .94, .04], [.0, .06, .78]]); c = fade(c, .08, .93)
    return contrast(sat(tone(c, sh=(.04, .02, -.02), hi=(.06, .04, -.03)), .9), .1)
def L_cross(c):
    r = clip(c[..., 0]); g = clip(c[..., 1]); b = clip(c[..., 2])
    o = np.stack([np.clip(r*1.08 + (r**2 - r)*.5, 0, 1), np.clip(g*1.0 + (g - g**2)*.15, 0, 1), np.clip(.1 + b*.7 + (b**2)*.1, 0, 1)], -1)
    return sat(contrast(o, .2), 1.15)
def L_matte(c):
    c = fade(contrast(c, .1), .09, .94); return sat(tone(c, sh=(.0, .01, .02), hi=(.01, .0, -.01)), .92)
def L_vintagefade(c):
    c = fade(c, .12, .9); c = mix(c, [[.95, .06, .0], [.02, .92, .04], [.0, .08, .84]])
    return sat(tone(c, sh=(.03, .03, .05), hi=(.05, .03, -.04)), .85)
def L_softskin(c):
    c = hue_shift(c, 20, 35, 0, .88, 1.05); c = lgg(c, lift=.015, gamma=1.08, gain=1.0)
    return soft_clip(fade(contrast(c, -.15), .0, .98), .9)
def L_peach(c):
    c = mix(c, [[1.04, .04, -.02], [.0, .98, .02], [.0, .02, .94]]); c = tone(c, sh=(.04, .0, .01), hi=(.05, .02, -.01))
    return soft_clip(lgg(vib(c, .2), gamma=1.1), .9)
def L_glow(c):
    c = lgg(c, lift=.02, gamma=1.2, gain=1.0); c = tone(c, hi=(.05, .03, .0)); return soft_clip(sat(c, 1.05), .9)
def L_rosy(c):
    c = mix(c, [[1.0, .0, .0], [-.03, .97, .02], [.03, .0, 1.02]]); c = tone(c, sh=(.02, -.01, .03), hi=(.04, .0, .02))
    return soft_clip(vib(lgg(c, gamma=1.08), .15), .92)
def L_cleanbright(c):
    c = lgg(c, gamma=1.18); return soft_clip(contrast(vib(c, .25), .15), .92)
def L_fresh(c):
    c = hue_shift(c, 110, 50, -6, 1.2, 1.04); c = lgg(c, gamma=1.1); return soft_clip(contrast(vib(c, .3), .18), .92)
def L_vividpop(c):
    return soft_clip(contrast(sat(c, 1.35), .28), .92)
def L_tasty(c):
    c = hue_shift(c, 30, 35, -3, 1.25); c = hue_shift(c, 8, 25, 0, 1.15); c = tone(c, hi=(.05, .02, -.04))
    return soft_clip(contrast(vib(c, .3), .25), .9)
def L_cafe(c):
    c = mix(c, [[1.0, .06, .0], [.0, .94, .02], [.0, .04, .8]]); c = fade(c, .03, .96)
    return soft_clip(contrast(sat(c, .92), .25), .9)
def L_bw(c): return contrast(mono(c), .25)
def L_bwhigh(c): return contrast(mono(c, (.4, .5, .1)), .7)
def L_silver(c): return fade(contrast(mono(c), .15), .05, .95)
def L_sepia(c): return contrast(tint_mono(c, (.09, .05, .02), (1.0, .89, .7)), .1)
def L_cyan(c): return contrast(tint_mono(c, (.02, .08, .16), (.88, .98, 1.0)), .1)
def L_dreampink(c):
    c = fade(c, .06, .97); c = tone(c, sh=(.06, .0, .06), hi=(.06, .0, .05)); return soft_clip(lgg(sat(c, 1.1), gamma=1.12), .92)
def L_cyber(c):
    c = tone(c, sh=(.02, -.02, .12), hi=(.10, -.02, .08)); c = hue_shift(c, 200, 60, -12, 1.4); c = hue_shift(c, 25, 40, -6, 1.15)
    return contrast(sat(c, 1.3), .4)
def L_sunset(c):
    c = mix(c, [[1.1, .05, -.04], [.0, .92, .0], [-.04, .06, .88]]); c = tone(c, sh=(.0, -.01, .06), hi=(.1, .03, -.06))
    return soft_clip(contrast(vib(c, .3), .25), .9)
def L_nightblue(c):
    c = mix(c, [[.85, .04, .05], [.0, .92, .08], [.02, .04, 1.1]]); c = lgg(c, gamma=.92); return contrast(tone(sat(c, .9), sh=(-.02, .0, .06)), .3)
def L_forest(c):
    c = hue_shift(c, 100, 60, -8, 1.15, .94); c = mix(c, [[.95, .03, .02], [.0, 1.0, .0], [.0, .04, .94]]); return contrast(sat(c, .95), .25)
def L_autumn(c):
    c = hue_shift(c, 95, 55, -28, 1.1); c = tone(c, hi=(.07, .03, -.06)); return soft_clip(contrast(vib(c, .2), .2), .9)
def L_winter(c):
    c = mix(c, [[.94, .0, .06], [.0, .98, .04], [.02, .04, 1.08]]); c = lgg(c, lift=.01, gamma=1.06); return soft_clip(contrast(sat(c, .88), .2), .93)
def L_hdr(c):
    c = lgg(c, gamma=1.22); return soft_clip(contrast(vib(sat(c, 1.2), .3), .5), .9)
def L_lowcon(c): return fade(contrast(c, -.3), .06, .96)

LOOKS = [
 # id, label, category, function
 ('blockbuster', 'Blockbuster', 'Cinematic', L_blockbuster), ('tealorange', 'Teal Orange', 'Cinematic', L_tealorange),
 ('moodyteal', 'Moody Teal', 'Cinematic', L_moodyteal), ('goldenhour', 'Golden Hour', 'Cinematic', L_goldenhour),
 ('bleach', 'Bleach Bypass', 'Cinematic', L_bleach), ('hdr', 'HDR Punch', 'Cinematic', L_hdr),
 ('portra', 'Portra', 'Film', L_portra), ('fujigreen', 'Fuji Green', 'Film', L_fujigreen),
 ('seventies', '70s', 'Film', L_70s), ('cross', 'Cross Process', 'Film', L_cross),
 ('matte', 'Matte', 'Film', L_matte), ('vintagefade', 'Vintage Fade', 'Film', L_vintagefade),
 ('softskin', 'Soft Skin', 'Portrait', L_softskin), ('peach', 'Peach', 'Portrait', L_peach),
 ('glow', 'Glow', 'Portrait', L_glow), ('rosy', 'Rosy', 'Portrait', L_rosy), ('cleanbright', 'Clean Bright', 'Portrait', L_cleanbright),
 ('fresh', 'Fresh', 'Life', L_fresh), ('vividpop', 'Vivid Pop', 'Life', L_vividpop),
 ('tasty', 'Tasty', 'Life', L_tasty), ('cafe', 'Cafe', 'Life', L_cafe), ('lowcon', 'Low Contrast', 'Life', L_lowcon),
 ('bw', 'B&W', 'Mono', L_bw), ('bwhigh', 'B&W High', 'Mono', L_bwhigh), ('silver', 'Silver', 'Mono', L_silver),
 ('sepia', 'Sepia', 'Mono', L_sepia), ('cyanotype', 'Cyanotype', 'Mono', L_cyan), ('neonoir', 'Neo Noir', 'Mono', L_neonoir),
 ('dreampink', 'Dream Pink', 'Mood', L_dreampink), ('cyber', 'Cyberpunk', 'Mood', L_cyber), ('sunset', 'Sunset', 'Mood', L_sunset),
 ('nightblue', 'Night Blue', 'Mood', L_nightblue), ('forest', 'Forest', 'Mood', L_forest), ('autumn', 'Autumn', 'Mood', L_autumn),
 ('winter', 'Winter', 'Mood', L_winter),
]

def identity():
    t = (np.arange(N)/(N - 1.0))
    b, g, r = np.meshgrid(t, t, t, indexing='ij')   # axes: [blue][green][red]
    return np.stack([r, g, b], -1)

def strip_png(rgb):
    # rgb[b][g][r] -> image width = N*N (x = b*N + r), height = N (y = g)
    a = np.clip(np.round(rgb*255), 0, 255).astype(np.uint8)
    img = np.transpose(a, (1, 0, 2, 3)).reshape(N, N*N, 3)
    bio = io.BytesIO(); Image.fromarray(img, 'RGB').save(bio, 'PNG', optimize=True); return bio.getvalue()

def main():
    os.makedirs(OUT, exist_ok=True)
    idg = identity(); manifest = []
    open(os.path.join(OUT, '_identity.png'), 'wb').write(strip_png(idg))
    for lid, label, cat, fn in LOOKS:
        out = clip(fn(idg.copy()))
        png = strip_png(out)
        js = "NFLooks.add(%s);\n" % json.dumps({'id': lid, 'size': N, 'png': 'data:image/png;base64,' + base64.b64encode(png).decode()})
        open(os.path.join(OUT, lid + '.js'), 'w').write(js)
        open(os.path.join(OUT, lid + '.png'), 'wb').write(png)   # plain LUT strip too (open it in any editor)
        manifest.append([lid, label, cat])
    open(os.path.join(HERE, 'looks.js'), 'w').write('/* generated by generate_looks.py: [id, label, category]. Each look lives in looks/<id>.js (+ looks/<id>.png strip) */\nwindow.NF_LOOK_LIST=' + json.dumps(manifest) + ';\n')
    print(len(manifest), 'looks written')
if __name__ == '__main__': main()
