import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import nets,tool,Net_Model
from torchvision import transforms
import time
import cv2


class Detector:
    def __init__(self, pnet_param="./temps/p_net.pth", rnet_param="./temps/R_net.pth", onet_param="./temps/O_net.pth"):

        self.pnet = Net_Model.PNet()
        self.rnet = Net_Model.RNet()
        self.onet = Net_Model.ONet()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pnet.to(self.device)
        self.rnet.to(self.device)
        self.onet.to(self.device)


        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect(self, image):
        start_time = time.time()
        pnet_boxes = np.array(self.__pnet_detect(image))
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = np.array(self.__rnet_detect(image, pnet_boxes))
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = np.array(self.__onet_detect(image, rnet_boxes))
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __pnet_detect(self, img):

        boxes = []
        w, h = img.size
        # print(w,h)
        min_side_len = min(w, h)

        scale = 1

        while min_side_len >= 12:
            img_data = self.__image_transform(img).to(self.device)

            # img_data = torch.unsqueeze(img_data, dim=0)  # 扩维度将[C,H,W]转为[N,C,H,W]
            img_data.unsqueeze_(0)

            _cls, _offest = self.pnet(img_data)
            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            # print(cls.shape) # w h
            # print(offest.shape) #4 w h

            idxs = torch.nonzero(torch.gt(cls, 0.6))
            # print(idxs.shape)
            boxes.extend(self.__box(idxs, offest, cls[idxs[:, 0], idxs[:, 1]], scale))

            scale *= 0.707
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)
        return tool.nms(np.array(boxes), 0.3)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = np.round((start_index[:, 1] * stride) / scale)  # 宽，W，x
        _y1 = np.round((start_index[:, 0] * stride) / scale)  # 高，H,y
        _x2 = np.round((start_index[:, 1] * stride + side_len) / scale)
        _y2 = np.round((start_index[:, 0] * stride + side_len) / scale)

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[:, 0], start_index[:, 1]]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        box = [x1.numpy(), y1.numpy(), x2.numpy(), y2.numpy(), cls.numpy()]
        box = np.array(box).T

        return box

    def __rnet_detect(self, image, pnet_boxes):
        _img_dataset = []
        _pnet_boxes = tool.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset=torch.stack(_img_dataset)
        img_dataset=img_dataset.to(self.device)

        _cls, _offset = self.rnet(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        idxs, _ = np.where(_cls > 0.7)
        _box = _pnet_boxes[idxs]
        _x1 = (_box[:, 0])
        _y1 = (_box[:, 1])
        _x2 = (_box[:, 2])
        _y2 = (_box[:, 3])

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]
        cls = _cls[idxs][:, 0]
        boxes = [x1, y1, x2, y2, cls]
        boxes = tool.nms(np.array(boxes).T, 0.3)

        return boxes

    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = tool.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _offset = self.onet(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        idxs, _ = np.where(_cls > 0.97)
        _box = _rnet_boxes[idxs]
        _x1 = (_box[:, 0])
        _y1 = (_box[:, 1])
        _x2 = (_box[:, 2])
        _y2 = (_box[:, 3])
        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]
        cls = _cls[idxs][:, 0]
        boxes = [x1, y1, x2, y2, cls]
        boxes = tool.nms(np.array(boxes).T, 0.3,isMin=True)

        return boxes
# if __name__ == '__main__':
#     x = time.time()
#     with torch.no_grad() as grad:
#         image_file = r"MTCNN作业附件图片1/10.jpg"
#         detector = Detector()
#
#         with Image.open(image_file) as im:
#             boxes = detector.detect(im)
#             imDraw = ImageDraw.Draw(im)
#             for box in boxes:
#                 x1 = int(box[0])
#                 y1 = int(box[1])
#                 x2 = int(box[2])
#                 y2 = int(box[3])
#                 # print(box[4])
#                 print((x1, y1, x2, y2))
#                 imDraw.rectangle((x1, y1, x2, y2), outline='red',width=3)
#             y = time.time()
#             print(y - x)
#             im.show()

if __name__ == '__main__':

    with torch.no_grad() as grad:
        # path = r"100.mp4"
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture(path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # w = int(cap.get(3))
        # h = int(cap.get(4))
        #
        # fourc = cv2.VideoWriter_fourcc(*"mp4v")
        # out = cv2.VideoWriter("02.mp4", fourc, fps, (w, h))
        while True:
            ret, frame = cap.read()
            # print(type(frame))
            x = time.time()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif ret == False:
                break

            detector = Detector()


            im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            boxes = detector.detect(im)
            imDraw = ImageDraw.Draw(im)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                # print(box[4])
                # print((x1, y1, x2, y2))
                imDraw.rectangle((x1, y1, x2, y2), outline=(235,61,131),width=3)
            y = time.time()
            z = y - x
            print("fps:", torch.true_divide(1,z))
            img = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
            cv2.imshow("", img)

        cap.release()
        cv2.destroyAllWindows()
