import Net_Model,Net_Train
import os
if __name__ == '__main__':
    net = Net_Model.ONet()
    if not os.path.exists("./param0"):
        os.makedirs("./param0")
    trainer = Net_Train.Trainer(net, './param0/o_net.pth', r"D:\PycharmProject\MTCNN\face_detection\celeba\48",
                                     r"D:\PycharmProject\MTCNN\face_detection\celeba_validate\48")
    trainer.trainer()