import jieba
import wordcloud
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt

py = imageio.imread('cryc.png')
f = open('text.txt', encoding='utf-8')
txt = f.read()
txt_list = jieba.lcut(txt)
string = ' '.join(txt_list)
# 词云图设置
wc = wordcloud.WordCloud(
    width=2180,         # 图片的宽
    height=2755,         # 图片的高
    background_color='white',   # 图片背景颜色
    font_path='msyh.ttc',    # 词云字体
    mask=py,     # 所使用的词云图片
    scale=15,
    stopwords={'女生'},         # 停用词
    # contour_width=5,
    # contour_color='red'  # 轮廓颜色
)
# 给词云输入文字
wc.generate(string)
plt.imshow(wc)
plt.show()
# 词云图保存图片地址
wc.to_file('out.png')