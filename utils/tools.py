import os
import termcolor
import cv2
import numpy as np
import pdb

# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    
    if not os.path.exists(logdir):
       os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
   # print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print(toCyan("Resume model parameters from {}".format(ckpt_path)))


def read_tri_image_by_index(index,img_dir,h=128,w=128):

    p1=os.path.join(img_dir,'08',str(index)+'.jpg')
    p2=p1.replace('/08/','/10/')
    p3=p1.replace('/08/','/14/')

    img1=cv2.imread(p1,0)
    img2=cv2.imread(p2,0)
    img3=cv2.imread(p3,0)
    imgs_=[img1,img2,img3]
    imgs=[np.expand_dims(x,-1) for x in imgs_]
    img=np.concatenate(imgs,-1)
    img=img.astype(np.float32)
    img=cv2.resize(img,(h,w))
    img=img/255.
    return np.expand_dims(img,0)


def calc_score(f_gt,f_pred):
    # s1
    gt=[]
    pred=[]
    for line1,line2 in zip(f_gt.readlines(),f_pred.readlines()):
        ss1=line1.strip().split(' ')
        ss2=line2.strip().split(' ')
        gt.append([int(ss1[0]),np.float32(ss1[1]),np.float32(ss1[2])])
        pred.append([int(ss2[0]),np.float32(ss2[1]),np.float32(ss2[2])])

    gt=np.array(gt)
    pred=np.array(pred)
    # pdb.set_trace()
    a=np.sum(pred[gt[:,0]==1,0]==1)
    b=np.sum(pred[gt[:,0]==1,0]==-1)
    c=np.sum(pred[gt[:,0]==-1,0]==1)
    d=np.sum(pred[gt[:,0]==-1,0]==-1)
    e=(a+b)*(a+c)/(a+b+c+d)
    ets=(a-e)/(a+b+c-e)
    if ets<0.1:
        s1=0
    elif ets>=0.1 and ets<0.2:
        s1=20
    elif ets>=0.2 and ets<0.3:
        s1=40
    elif ets>=0.3 and ets<0.4:
        s1=60
    elif ets>=0.4:
        s1=80

    # s2
    index=gt[:,0]==1
    pred=pred[index,:]
    gt=gt[index,:]
    index=pred[:,0]==1
    pred=pred[index,:]
    gt=gt[index,:]
    l2=np.sum(np.square(pred-gt),1)
    mae=np.mean(np.sqrt(l2))

    if mae>20:
        s2=0
    elif mae<=20:
        s2=20-mae
    s=s1+s2
    
    return s1,s2,s,ets,mae

def parse_heatmap(heatmap,ths):
    heatmap=np.squeeze(heatmap)
    height,width=heatmap.shape
    img_=(heatmap-np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))
    img_=img_*255
    img_=np.array([img_,img_,img_]).transpose((1,2,0))
    img=img_.astype(np.uint8).copy()

    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 255*ths, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # pdb.set_trace()
    # print(toMagenta(len(contours)))
    points=[]
    for c in contours:
      # find bounding box coordinates
      x,y,w,h = cv2.boundingRect(c)
      cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
      # find minimum area
      rect = cv2.minAreaRect(c)
      # calculate coordinates of the minimum area rectangle
      box = cv2.boxPoints(rect)
      # normalize coordinates to integers
      box = np.int0(box)
      # draw contours
      cv2.drawContours(img, [box], 0, (0,0, 255), 3)
      
      # calculate center and radius of minimum enclosing circle
      (x,y),radius = cv2.minEnclosingCircle(c)
      # cast to integers
      center = (int(x),int(y))
      radius = int(radius)

      # print(toMagenta(center))
      # print(toMagenta(radius))
      if radius>5:
        points.append([x,y])
      # draw the circle
      img = cv2.circle(img,center,radius,(0,255,0),2)
    
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    if len(points)>0:
        xy=1
        x=np.mean(np.array(points),0)[0]*800/height
        y=np.mean(np.array(points),0)[1]*800/width
    else:
        xy=-1
        x=-1
        y=-1
    txtline=str(xy)+' '+str(y)+' '+str(x)+'\n'
    
    return img, txtline
