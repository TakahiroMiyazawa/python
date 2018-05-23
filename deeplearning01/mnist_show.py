import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
"""
mnistデータの読み込みには3つの引数を設定できる
normalize  /入力画像を0.0~1.0の値に正規化(Falseだと0~255のピクセル)
flatten　　/入力画像を1次元配列にする(Falseだと1x28x28の三次元配列)
one_hot_label  /one-hot表現,正解となるラベルだけが１(Falseだと5のように正解ラベルが単純に格納)
"""



img = x_train[0] #訓練画像の0番めを格納
label = t_train[0] #0番めのラベル
print(label)  #5の画像なのでラベルは5

print(img.shape) #(784,)が表示
#flattenがTrueなので読み込んだ画像はNumPy配列として1次元で格納されている(784)
img = img.reshape(28, 28) #形状を元の画像サイズに変形
print(img.shape) #(28,28)が表示

img_show(img)
