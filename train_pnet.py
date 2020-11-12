import Net_Model,Net_Train
import os
if __name__ == '__main__':
    net = Net_Model.PNet()
    if not os.path.exists("./param0"):
        os.makedirs("./param0")
    trainer = Net_Train.Trainer(net, './param0/p_net.pth', r"D:\PycharmProject\MTCNN\face_detection\celeba\12",
                                     r"D:\PycharmProject\MTCNN\face_detection\celeba_validate\12")
    trainer.trainer()
    #最终偏移量的R2为0.5973124143653568