import gc

import cv2
import numpy as np
from src.Image.camera import Camera
from timeit import default_timer as timer


# bgLearn
# Description:
# Learn the backgroud by pics from cam then get a background model
# bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

#背景学习的功能就是事先采集多张图片来学习当前的背景信息，得到当前的背景模型，然后在实际使用中时候，每当采集一张图片，就可以
#通过背景模型来去掉该图片的背景。
#具体测试例子见本文件最后的main函数，首先通过bgobj = Bglearn(N)来得到背景学习对象，传入的参数N为采集N张图片来进行背景学习;
#通过 bgobj.studyBackgroundFromCam(cam)方法来学习背景;
#通过 bgobj.createModelsfromStats(N）方法来计算得到背景模型;传入的参数N为背景上下阈值之间的区分度
#通过frameDelBg = bgobj.delBg（frame）方法来将当前帧frame去掉背景，返回去掉背景后的图像frameDelBg.


class Bglearn():
#background learn method class#

    def __init__(self,bgStudyNum):#初始化函数，定义一些需要用到的参数
        """

        :param bgStudyNum:背景图像的数量，就是需要从多少张图片中去学习背景建模
        """
        #private attribute of class
        # how many pics captured for background study
        self.BG_STUDY_NUM = bgStudyNum
        # a list for store the pics captqured waited for study
        #bgVector是一个用于临时存储背景图像的向量，使用完需要释放此向量。
        #图像的宽高为960*1280，由于是RGB图像，所以维度为3
        self.bgVector = np.zeros(shape=(self.BG_STUDY_NUM, 960, 1280, 3), dtype=np.float32)
        # average of frames
        self.IavgF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)#IavgF是多帧数平均后的图像
        # pre frame
        self.IprevF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)#IprevF是前一帧的图像
        # temp frame for calculate
        self.Iscratch2 = np.zeros(shape=(960, 1280, 3), dtype=np.float32)#当前帧与前一帧的帧差：Iscratch2=I-IprevF
        # average difference of frames
        self.IdiffF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)#IdiffF计算一定时间内每帧图像之差
        # high threshold of background
        self.IhiF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)#高阈值的背景图
        # low threshold of background
        self.IlowF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)#低阈值的背景图
        # statistic the count of pics captured waited for study
        self.Icount = 0                                 #统计每一秒有多少帧，初始化为0
        # statistic the Frame number Pre One Second
        #self.nFrameNumPreOneSec = 0

        # kernel for image morph process
        self.kernel5 = np.ones((5, 5), np.uint8)
        self.kernel7 = np.ones((7, 7), np.uint8)
        self.kernel13 = np.ones((13, 13), np.uint8)#这些都是用作形态学处理的核
        self.kernel19 = np.ones((19, 19), np.uint8)
        self.kernel25 = np.ones((25, 25), np.uint8)

        # for show
        self.show = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)#用于展示的帧

        # time cost of one frame delete the background
        self.bgTimeCost=0


    def avgBackground(self, I):
        # read background pic of I,and then calculate frame difference of current frame and pre frame: I and IprevF
        # accumulate every frame difference Iscratch2 to sum of differences :IdiffF
        # meanwhile,accumulate every frame I  to sum of frames :IavgF

        """
        Parameters
        --------------
        I: input  Mat type pic stream

        Returns
        -------

        Examples
        --------
        """
        #学习平均背景步骤：
        # 1，使用cv2.accumulate函数累积每一参与学习的帧 ，之后会根据帧数做平均(IavgF)
        # 2，使用cv2.absdiff函数获取当前帧(I)与前一帧(IprevF)的帧差(Iscratch2)
        # 3,再次使用cv2.accumulate函数对帧差(Iscratch2)做累积，之后会根据帧数(Icount)做平均得到(IdiffF)
        cv2.accumulate(I, self.IavgF)
        # cv2.absdiff(I,IprevF, Iscratch2)
        self.Iscratch2 = cv2.absdiff(I, self.IprevF)
        cv2.accumulate(self.Iscratch2, self.IdiffF)

        # print("IavgF[100,100,0]:", self.IavgF[100, 100, 0])
        # print("IdiffF[100,100,0]:", self.IdiffF[100, 100, 0])
        self.Icount += 1.0
        self.IprevF = I.copy()

    def createModelsfromStats(self, scale):
        # calculate the average sum of frames to  IavgF
        # calculate frame difference to IdiffF
        # then  multiply the scale to the Idiff,and add the Idiff to IavgF to get the IhiF,
        # subtract  the Idiff from IavgF to get the IlowF
        # now we get the background model IhiF and IlowF
        """
           Parameters
           --------------
           scale:   gap of high threshold and low threshold of background model

           Returns
           -------

           Examples
           --------
           """

        # print("Icount", self.Icount)
        # Icount+=1
        self.IavgF = self.IavgF/self.Icount
        self.IdiffF = self.IdiffF/self.Icount
        # print("IavgF[100,100,0]:", self.IavgF[100, 100, 0])
        # print("IdiffF[100,100,0]:", self.IdiffF[100, 100, 0])
        self.IdiffF = cv2.add(self.IdiffF, 1.0)
        # cv2.imshow("avg", IavgF)
        # cv2.imshow("diff", IdiffF)
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\a.jpg", self.IavgF)
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\d.jpg", self.IdiffF)
        # cv2.add(IavgF,IdiffF, IhiF)

        # 判断是否为前景，需要两个最低和最高的阈值。
        # 如果大于高阈值或者小于低阈值，那么可以判断为前景，否则，依然为背景 而获取
        self.IdiffF = self.IdiffF * scale
        # print("IdiffF[mod:", self.IdiffF[100, 100, 0])
        self.IhiF = cv2.add(self.IavgF, self.IdiffF)
        # cv2.subtract(IavgF, IdiffF, IlowF)
        self.IlowF = cv2.subtract(self.IavgF, self.IdiffF)
        #release the memory | 12.25 内存优化，显示释放内存
        del self.bgVector
        gc.collect()
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\h.jpg", self.IhiF)
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\l.jpg", self.IlowF)

    def studyBackgroundFromCam(self, cam):
        # get many pics for time interval of 60sec by cam and store the pics in  bgVector.
        #then  call the avgBackground method
        """
            Parameters
             --------------
             cam: input camera object
               Returns
            -------
               Examples
            --------
            """
        try:
            #set the loop break condition over_flag,when pics waited for study is captured enough,the flag will be changed
            over_flag = 1
            pic_cnt = 0
            while (over_flag):
                # get image from camera
                frame, nFrameNum, t = cam.getImage()
                fin = np.float32(frame)
                # print("shape", fin.shape)
                # store the frame in list bgVector
                self.bgVector[pic_cnt] = fin
                # print("pic_cnt", pic_cnt)
                # wait about 200 milli seconds
                cv2.waitKey(200)
                pic_cnt += 1
                # print("pic_cnt", pic_cnt)
                if (pic_cnt == self.BG_STUDY_NUM):
                    over_flag = 0

            # print("shapebg", self.bgVector.shape)
            for i in range(self.bgVector.shape[0]):
                # print("i", i)
                self.avgBackground(self.bgVector[i])
        except Exception as e:
            print(e)
            #when occur exception ,the camera will disconnect
            cam.destroy()


    def backgroundDiff(self, src0, dst):
        # when get pic frame from camera, use the backgroundDiff to  segment the frame pic and get a mask pic
        # if the pic pixel value is higher than  high background threadhold  or lower than low background threadhold, the pixels
        # will change to white,otherwise, it will cover to black.
        # https://www.cnblogs.com/mrfri/p/8550328.html
        # rectArray=np.zeros(shape=(1,4),dtype=float)

        """
               Parameters
               --------------
               src0:      input cam pic waited for segment
               dst:       temp store segment result of mask

               Returns
               rectArray,   all boundingboxes of all bottles
               dst         segment result of mask
               -------
               Examples
               --------
           """

        rectArray = []
        src = np.float32(src0)
        # print("IlowF.shape", self.IlowF.shape)
        # print("IhiF.shape", self.IhiF.shape)
        # print("src.shape", src.shape)
        # print("dst.shape", dst.shape)

        # print("IlowF.tpye", self.IlowF.dtype)
        # print("IhiF.tpye", self.IhiF.dtype)
        # print("src.tpye", src.dtype)
        # print("dst.tpye", dst.dtype)

        # cv2.inRange(src, IlowF, IhiF, dst)
        #segment the src through IlowF and IhiF
        dst = cv2.inRange(src, self.IlowF, self.IhiF)
        #cv2.imshow("segment_debug", dst)

        #morph process the frame to clear the noise and highlight our object region
        # print("is this ok00?")
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.kernel7)
        # print("is this ok01?")
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel7)
        # print("is this ok02?")

        tmp = 255 * np.ones(shape=dst.shape, dtype=dst.dtype)
        # np.zeros(shape=(960, 1280, 3), dtype=np.uint8)

        #inverse the  pixel value to make the mask
        dst = cv2.subtract(255, dst)
        # dst=tmp-dst
        # print("is this ok03?")
        # cv2.GaussianBlur(dst, dst, (19, 19), 3)

        #filter  and morph again and then find the bottle contours
        dst = cv2.GaussianBlur(dst, (19, 19), 3)
        # print("is this ok04?")
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel13)  # eclipice
        dst = cv2.erode(dst, self.kernel19)  # eclipice
        # 解决cv2版本3.4.2和4.1.2兼容问题
        if cv2.__version__.startswith("3"):
            _, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        elif cv2.__version__.startswith("4"):
            contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.show, contours, -1, (0, 255, 0), 3)
        contourLen = len(contours)
        # print(contourLen)
        momentList = []
        pointList = []
        if contourLen > 0:  # range create a list of interger ,useful in loop
            for i in range(contourLen):

                #calculate all features of contours and draw rectangle of bounding box of contour
                contourM = cv2.moments(contours[i])  # every contour's  moment
                contourCenterGx = int(contourM['m10'] / contourM['m00'])# 重心
                contourCenterGy = int(contourM['m01'] / contourM['m00'])
                contourArea = cv2.contourArea(contours[i])# 面积
                contourhull = cv2.convexHull(contours[i])  # 凸包
                cv2.polylines(self.show, [contourhull], True, (500, 255, 0), 2)

                #https: // blog.csdn.net / lanyuelvyun / article / details / 76614872
                rotateRect = cv2.minAreaRect(contours[i]) #旋转外接矩形
                angle = rotateRect[2]
                diameter = min(rotateRect[1][0], rotateRect[1][1])


                contourBndBox = cv2.boundingRect(contours[i])  # x,y,w,h  外接矩形
                # print("contourBndBox type", type(contourBndBox))
                x = contourBndBox[0]
                y = contourBndBox[1]
                w = contourBndBox[2]
                h = contourBndBox[3]
                img = cv2.rectangle(self.show, (x, y), (x + w, y + h), (0, 255, 0), 2)  #画矩形
                rows, cols = src.shape[:2]  # shape 0 1 #得出原图的行 列 数
                [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01) #对轮廓进行多边形拟合
                lefty = int((x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                # print("pre final")
                # rectArray = np.append(rectArray, contourBndBox, axis=0)
                rectArray.append(contourBndBox) #存轮廓信息到数组中

                # print("final")
                # print("rectArray", rectArray)
        return rectArray, dst



    def delBg(self,src):
        # use the mask pic bgMask to make bit and operation to the cam frame to get a pic that del the bacground
        """
               Parameters
               --------------
               src:      input cam pic waited for segment
               dst:       temp store segment result of mask

               Returns
               rectArray,   all boundingboxes of all bottles
               dst         segment result of mask
               -------
               Examples
               --------
           """
        prev_time = timer()
        #simply output the frame that delete the background
        #dst = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)  flexible the dst shape to adapt the src shape
        dst = np.zeros(shape=src.shape, dtype=src.dtype)
        resarray, bgMask = self.backgroundDiff(src, dst)   #对src 帧 减去背景 结果放到dst，获得瓶子的框，和掩膜图像
        #bit and operation
        frame_delimite_bac = cv2.bitwise_and(src, src, mask=bgMask)#用掩膜图像和原图像做像素与操作，获得只有瓶子的图
        curr_time = timer()
        #calculate the cost time
        exec_time = curr_time - prev_time
        self.bgTimeCost =exec_time
        # print("Del background Cost time:", self.bgTimeCost)
        return frame_delimite_bac, bgMask



if __name__=="__main__":
#test case
    cam = Camera()
    # nConnectionNum = cam.get_device_num()
    #_data_buf, _nPayloadSize = cam.connectCam()

#get frames from cam and learn the background model
    bgobj = Bglearn(50)
    bgobj.studyBackgroundFromCam(cam)
    bgobj.createModelsfromStats(6.0)

    while 1:
        try:
            # get frames from cam realtime
            frame, nFrameNum, t = cam.getImage()
            """
            # init the fps calculate module
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:      # time exceed 1 sec and we will update values
                fpsnum = nFrameNum - bgobj.nFramNumPreOneSec
                # print("fpsnum",fpsnum)
                fps = "FPS: " + str(fpsnum)
                bgobj.nFrameNumPreOneSec = nFrameNum    #update the nFrameNumPreOneSec every 1 second
                accum_time =0                           #back to origin
            """
            cv2.imshow("cam", frame)
            # copy a frame for show
            bgobj.show = frame.copy()
            # get fps of cam output
            fps = cam.getCamFps(nFrameNum)
            # use the background model to del the bacground of c    qqqam frame
            frameDelBg = bgobj.delBg(frame)
            # put text on frame to display the fps
            cv2.putText(frameDelBg, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.imshow("output", frameDelBg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(10)
        except Exception as e:
            # print(e)
            cam.destroy()
    cam.destroy()


    # checkImage

    # Description:
    # checkImage is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

    # checkImage Implemente Details:

