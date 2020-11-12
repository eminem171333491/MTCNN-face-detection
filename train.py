import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from sampling import Mydate
from sklearn.metrics import r2_score
# import time
# from earlystopping import EarlyStopping

class Trainer:
    def __init__(self, net, save_path, train_dataset_path, validate_dataset_path):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = net.to(self.device)
        self.save_path = save_path
        self.train_dataset_path = train_dataset_path
        self.validate_dataset_path = validate_dataset_path
        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        self.offset_r2 = 0
        self.count = 0

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
        else:
            print("NO Param")

    def trainer(self):
        train_faceDataset = Mydate(self.train_dataset_path)
        train_dataloader = DataLoader(train_faceDataset, batch_size=512, shuffle=True, pin_memory=True,num_workers=8)
        validate_faceDataset = Mydate(self.validate_dataset_path)
        validate_dataloader = DataLoader(validate_faceDataset, batch_size=512, shuffle=True, pin_memory=True,num_workers=8)
        v_offset_r2 = 0
        epoch = 1
        self.net.train()
        while True:
            label_offset = []
            label_cls = []
            out_offset = []
            out_cls = []
            for i, (img_data_, category_, offset_) in enumerate(train_dataloader):
                img_data_ = img_data_.to(self.device)
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)
                output_offset = _output_offset.view(-1, 4)

                category_mask = torch.lt(category_, 2)
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cls_loss_fn(output_category, category)

                offset_mask = torch.gt(category_, 0)
                offset = torch.masked_select(offset_, offset_mask)
                output_offset = torch.masked_select(output_offset, offset_mask)
                offset_loss = self.offset_loss_fn(output_offset, offset)

                loss = cls_loss + offset_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()

                label_offset.extend(offset.data.cpu().numpy())
                label_cls.extend(category.data.cpu().numpy())
                out_offset.extend(output_offset.data.cpu().numpy())
                out_cls.extend(output_category.data.cpu().numpy())

                offset_r2 = r2_score(label_offset, out_offset)
                cls_r2 = r2_score(label_cls, out_cls)

                print("epoch:",epoch," 训练集 loss:", loss, ", cls_loss:", cls_loss, ", offset_loss", offset_loss)
                print("epoch:",epoch," 置信度的R2:", cls_r2, ", 偏移量的R2:", offset_r2)

            for v_img_data_, v_category_, v_offset_ in validate_dataloader:
                v_img_data_ = v_img_data_.to(self.device)
                v_category_ = v_category_.to(self.device)
                v_offset_ = v_offset_.to(self.device)

                v_output_category, v_output_offset = self.net(v_img_data_)
                v_output_category_ = v_output_category.view(-1, 1)
                v_output_offset_ = v_output_offset.view(-1, 4)

                v_category_mask = torch.lt(v_category_, 2)
                v_category = torch.masked_select(v_category_, v_category_mask)
                v_output_category = torch.masked_select(v_output_category_, v_category_mask)
                v_cls_loss = self.cls_loss_fn(v_output_category, v_category)

                v_offset_mask = torch.gt(v_category_, 0)
                v_offset = torch.masked_select(v_offset_, v_offset_mask)
                v_output_offset = torch.masked_select(v_output_offset_, v_offset_mask)
                v_offset_loss = self.offset_loss_fn(v_output_offset, v_offset)

                loss = v_cls_loss + v_offset_loss

                v_cls_loss = v_cls_loss.cpu().item()
                v_offset_loss = v_offset_loss.cpu().item()
                v_loss = loss.cpu().item()

                label_offset.extend(v_offset.data.cpu().numpy())
                label_cls.extend(v_category.data.cpu().numpy())
                out_offset.extend(v_output_offset.data.cpu().numpy())
                out_cls.extend(v_output_category.data.cpu().numpy())

                v_offset_r2 = r2_score(label_offset, out_offset)
                cls_r2 = r2_score(label_cls, out_cls)

                print("epoch:",epoch," 验证集 loss:", v_loss, ", cls_loss:", v_cls_loss, ", offset_loss", v_offset_loss)
                print("epoch:",epoch," 置信度的R2:", cls_r2, ", 偏移量的R2:", v_offset_r2)

            if v_offset_r2 >= self.offset_r2:
                torch.save(self.net.state_dict(), self.save_path)
                print("第{0}轮参数保存成功，偏移量的R2由{1}更新至{2}".format(epoch, self.offset_r2, v_offset_r2))
                self.offset_r2 = v_offset_r2
                self.count = 0
                epoch += 1
            elif v_offset_r2 < self.offset_r2:
                self.count += 1
                print("第{0}轮参数未保存，已经{1}轮未更新".format(epoch,self.count))
                epoch += 1
                if self.count >= 10:
                    print("训练完成，本次共训练了{0}轮,最终偏移量的R2为{1}".format(epoch,self.offset_r2))
                    break

            # if epoch == 30:
            #     break
