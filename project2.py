#import datetime
import cv2
import sys
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import time
import os.path
import glob
import Tools
import math



def testIntersectionIn(x,y,k,b):
    if (np.abs(k*x-y+b)/math.sqrt(1+k*k)<15):
        return True
    return False


#def testIntersectionOut(x, y):
#    if ((y >= 595) and (y <= 605)):
#        return True
#    return False

start_time=time.time()
'''
1、将视频中出现的车辆裁剪出来，放到指定的目录
'''
videopath = sys.argv[1]
width = 1200
textIn = 0
textOut = 0
localtime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))#获得当前系统时间的字符串
output_dir = "/home/naruto/gcsj/output/"       #切割图片输出根路径
#创建存储图片的路径
if not os.path.exists(output_dir+localtime):
    os.mkdir(output_dir+localtime)
output_dir = output_dir+localtime	#切割图片输出路径
camera = cv2.VideoCapture(videopath)    #/home/naruto/Videos/video2.mp4
history = 200    # 训练帧数
#KNN
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # KNN背景建模
bs.setHistory(history)
frames = 0
r = 0.1
lines = Tools.Get_Road_Line(videopath,history,r)

# loop over the frames of the video
while True:
    # res:读取到帧
    res, frame = camera.read()
    if not res:
        break
    frames += 1
#    if frames < history:
#        continue
    if frames%1 != 0:
        continue 
    # 获取foreground mask
    fg_mask = bs.apply(frame)  
    #if frames < history:
     #   continue
    # 对原始帧进行膨胀去噪
    # 对帧进行二值化处理，阈值为0，最大值为225，阈值类型为THRESH_BINARY
    th = cv2.threshold(fg_mask.copy(), thresh=0, maxval=255, type=cv2.THRESH_BINARY)[1]
    #cv2.imshow("th1", th)
    #先腐蚀，消除细微颗粒点（3，3），迭代2次
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    #cv2.imshow("th", th)
    #再膨胀，填充前景（5，3），迭代3次
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=3)
    #cv2.imshow("dilated", dilated)
    # 获取所有检测框
    contours, hier = cv2.findContours(image=dilated,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (0, 400), (width, 400), (250, 0, 1), 2)  # blue line(图片、起点、终点、颜色、线条粗细)
    cv2.line(frame, (lines[0][0]+20, lines[0][1]), (lines[0][2]-20, lines[0][3]), (250, 0, 1), 2)  # blue line(图片、起点、终点、颜色、线条粗细)
    #cv2.line(frame, (lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]), (250, 250, 1), 2)  # line(图片、起点、终点、颜色、线条粗细)
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 10000:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)  #(x,y)左上角坐标(w,h):宽度和高度,boundingRect返回的是一个能够垂直包含该图像的最小矩形
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rectagleCenterPont = ((x + x + w) // 2, (y + y + h) // 2)   #矩形框中心点
        cv2.circle(frame, rectagleCenterPont, 1, (0, 0, 255), 5)
	#过标记线即统计车辆
        if (testIntersectionIn((x + x + w) // 2, (y + y + h) // 2,lines[2],lines[3])):
            textIn += 1
            cropImg = frame[y:y+h, x:x+w]  # 【行数据范围：列数据范围】
            cv2.imwrite(output_dir + '/' + str(textIn) + '.png', cropImg)
            #print(x, y, w, h)
        #if (testIntersectionOut((x + x + w) // 2, (y + y + h) // 2)):
        #    textOut += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    second_time = time.time()
    #cv2.putText(frame, "In: {}".format(str(textIn)), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.putText(frame, "Out: {}".format(str(textOut)), (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("frame", frame)

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()



'''
2、将output下的所有图片使用Alexnet网络测试
'''
caffe.set_device(0)
caffe.set_mode_gpu()
def UseAlexnet(img,labels):
    im=caffe.io.load_image(img) #加载图片
    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中
    #执行测试
    out = net.forward()    
    #prob= net.blobs['prob'].data[0].flatten()#取出最后一层（Softmax）属于某个类别的概率值
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1] 
    #for i in np.arange(top_k.size): 
        #print(top_k[i], labels[top_k[i]],prob[top_k[i]])	#类别、标签、置信度
    #print(labels[top_k[0]])
    return labels[top_k[0]]   
    
bus_count = 0
car_count = 0
motor_count = 0
truck_count = 0

caffe_root = '/home/naruto/.conda/envs/caffe/'
sys.path.insert(0,caffe_root+'python')
root = '/home/naruto/.conda/envs/caffe/'	#根目录
deploy = root + 'examples/imgsnet/bvlc_alexnet/deploy.prototxt'	#Alexnet网络配置文件
caffe_model = root + 'examples/imgsnet/bvlc_alexnet/caffe_alexnet_train_iter_250000.caffemodel'
labels_filename = root + 'examples/imgsnet/labels.txt'	#标签
mean_file = root + 'examples/imgsnet/ilsvrc_2012_mean.npy'	#均值文件
net = caffe.Net(deploy,caffe_model,caffe.TEST)	#加载model和Net
labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
#output_dir第一步中切割图片的路径
filename = os.listdir(output_dir)

#图片预处理
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})#设定图片shape格式
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值
transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR    
for img in filename:
    img = output_dir+'/'+img	#图片uri
    label = UseAlexnet(img,labels)	#使用Alexnet对每一张图片进行分类
    #print('img:',img)
    #print('label:',label)
    if label=='bus':
        bus_count = bus_count+1
    elif label=='car':
        car_count = car_count+1
    elif label=='motor':
        motor_count = motor_count+1
    elif label=='truck':
        truck_count = truck_count+1
    
#输出结果
print('bus:',bus_count)
print('car:',car_count)
print('motor:',motor_count)
print('truck:',truck_count)
'''
filepath = sys.argv[2]
f = open(filepath,'a')
f.write('bus:')
f.write(str(bus_count))
f.write('\ncar:')
f.write(str(car_count))
f.write('\nmotor:')
f.write(str(motor_count))
f.write('\ntruck:')
f.write(str(truck_count))
f.close()
'''
'''
3.img to video
'''
imgpath = output_dir + '/'
#print(imgpath)
filelist = os.listdir(imgpath)
filelist.sort()

fps = 24
if len(filelist)!=0:
    img = cv2.imread(imgpath+filelist[0])    
    size = (img.shape[1],img.shape[0])

 
    videoWriter = cv2.VideoWriter(imgpath+'video.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    #视频保存在当前目录下

    for item in filelist:
        if item.endswith('.png'): 
        #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            item = imgpath + item
            img = cv2.imread(item)
            videoWriter.write(img)

    end_time=time.time()
    print("pic process:", second_time-start_time)
    print(end_time-start_time)
    print(end_time-second_time)
    videoWriter.release()

cv2.destroyAllWindows()




