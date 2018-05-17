import numpy as np
import matplotlib.pyplot as plt

#データの作成
x=np.arange(0,6,0.1)#0から0.6まで0.1刻みで生成
y1=np.sin(x)
y2=np.cos(x)

#グラフの描画
plt.plot(x,y1,label="sin")
plt.plot(x,y2,label="cos",linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin&cos')#タイトル
plt.legend()#注釈を入れる
plt.show()

