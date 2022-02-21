from mymodel.common import Focus,Conv,C3,SPP,Concat
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, features, num_classes=80, init_weights=False):
        super().__init__()
        # Focus()
        # nn.Conv2d(ch_in, ch_out, kernel, stride)
        # Conv(ch_out, kernel, stride)
        # C3()
        # SPP()
        # Concat()怎么用？？？

        #backbone
        self.Focus = Focus(64,3)
        self.Conv_1 = Conv(128,3,2)
        self.C3_1 = C3(128) #3
        self.Conv_2 = Conv(256,3,2)
        self.C3_2 = C3(256) #9here
        self.Conv_3 = Conv(512,3,2)
        self.C3_3 = C3(512) #9here
        self.Conv_4 = Conv(1024,3,2)
        self.SPP = SPP(1024,5,9,13)
        self.C3_4 = C3(1024,False) #3

        #head
        self.Conv_5 = Conv(512,1,1)
        self.Upsample_1 = nn.Upsample(None,2,'nearest')
        #self.concat_1 = torch.cat((self.Upsample_1,self.C3_3),dim=1)
        #self.concat_1 = Concat((self.Upsample_1,self.C3_3),1)
        self.C3_5 = C3(512,False) #3

        self.Conv_6 = Conv(256,1,1)
        self.Upsample_2 = nn.Upsample(None,2,'nearest')
        #self.concat_2 = torch.cat((self.Upsample_2,self.C3_2),dim=1)
        #self.concat_2 = Concat((self.Upsample_2,self.C3_2),1)
        self.C3_6 = C3(256,False) #3

        self.Conv_7 = Conv(256,3,2)
        #self.concat_3 = torch.cat((self.Conv_7,self.Conv_6),dim=1)
        #self.concat_3 = Concat((self.Conv_7,self.Conv_6),1)
        self.C3_7 = C3(512,False) #3

        self.Conv_8 = Conv(256,3,2)
        #self.concat_4 = torch.cat((self.Conv_8,self.Conv_5),dim=1)
        #self.concat_4 = Concat((self.Conv_8,self.Conv_5),1)
        self.C3_8 = C3(256,False) #3

    def forward(self, x):
        x = self.Focus(x)
        x = self.Conv_1(x)
        x = self.C3_1(x)
        x = self.Conv_2(x)
        x = y2 = self.C3_2(x)
        x = self.Conv_3(x)
        x = y1 = self.C3_3(x)
        x = self.Conv_4(x)
        x = self.SPP(x)
        x = self.C3_4(x)
        x = y4 = self.Conv_5(x)
        x = self.Upsample_1(x)
        x = torch.cat((x,y1),dim=1)
        x = self.C3_5(x)
        x = y3 = self.Conv_6(x)
        x = self.Upsample_2(x)
        x = torch.cat((x,y2),dim=1)
        x = self.C3_6(x)
        x = self.Conv_7(x)
        x = torch.cat((x,y3),dim=1)
        x = self.C3_7(x)
        x = self.Conv_8(x)
        x = torch.cat((x,y4),dim=1)
        x = self.C3_8(x)
        return x
        # 能inplace? 应该只有部分块可以吧

class yolov5(nn.Module):
    def __init__(self, features, num_classes=80, init_weights=False):
        super().__init__()

        #backbone
        self.Focus = Focus(64,3)
        self.Conv_1 = Conv(128,3,2)
        self.C3_1 = C3(128) #3
        self.Conv_2 = Conv(256,3,2)
        self.C3_2 = C3(256) #9here
        self.Conv_3 = Conv(512,3,2)
        self.C3_3 = C3(512) #9here
        self.Conv_4 = Conv(1024,3,2)
        self.SPP = SPP(1024,5,9,13)
        self.C3_4 = C3(1024,False) #3

        #head
        self.Conv_5 = Conv(512,1,1)
        self.Upsample_1 = nn.Upsample(None,2,'nearest')
        self.C3_5 = C3(512,False) #3

        self.Conv_6 = Conv(256,1,1)
        self.Upsample_2 = nn.Upsample(None,2,'nearest')
        self.C3_6 = C3(256,False) #3

        self.Conv_7 = Conv(256,3,2)
        self.C3_7 = C3(512,False) #3

        self.Conv_8 = Conv(256,3,2)
        self.C3_8 = C3(256,False) #3

    def forward(self, x):
        x = self.Focus(x)
        x = self.Conv_1(x)
        x = self.C3_1(x)
        x = self.Conv_2(x)
        x = y2 = self.C3_2(x)
        x = self.Conv_3(x)
        x = y1 = self.C3_3(x)
        x = self.Conv_4(x)
        x = self.SPP(x)
        x = self.C3_4(x)
        x = y4 = self.Conv_5(x)
        x = self.Upsample_1(x)
        x = torch.cat((x,y1),dim=1)
        x = self.C3_5(x)
        x = y3 = self.Conv_6(x)
        x = self.Upsample_2(x)
        x = torch.cat((x,y2),dim=1)
        x = self.C3_6(x)
        x = self.Conv_7(x)
        x = torch.cat((x,y3),dim=1)
        x = self.C3_7(x)
        x = self.Conv_8(x)
        x = torch.cat((x,y4),dim=1)
        x = self.C3_8(x)
        return x
        # 可以这样inplace；根据self.save判断中间结果是否需要保存到y[]
    
    def forward_once(self, x):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                if isinstance(m.f, int) :
                    x = y[m.f]   # from earlier layers
                else :
                    for j in m.f :
                        if j != -1 : 
                            x.append(y[j]) 
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def forward_once(self, x):
        y = [] # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                if isinstance(m.f, int) :
                    x = y[m.f]   # from earlier layers
                else :
                    for j in m.f :
                        x.append(y[j])
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def forward(self, x):
        x = self.Focus(x) #0
        x = self.Conv_1(x) #1
        x = self.C3_1(x) #2
        x = self.Conv_2(x) #3
        x = y2 = self.C3_2(x) #4
        x = self.Conv_3(x) #5
        x = y1 = self.C3_3(x) #6
        x = self.Conv_4(x) #7
        x = self.SPP(x) #8?
        x = self.C3_4(x) #9?
        x = y4 = self.Conv_5(x) #10
        x = self.Upsample_1(x) #11
        x = torch.cat((x,y1),dim=1) #12
        x = self.C3_5(x) #13
        x = y3 = self.Conv_6(x) #14
        x = self.Upsample_2(x) #15
        x = torch.cat((x,y2),dim=1) #16
        x = self.C3_6(x) #17
        x = self.Conv_7(x) #18
        x = torch.cat((x,y3),dim=1) #19
        x = self.C3_7(x) #20
        x = self.Conv_8(x) #21
        x = torch.cat((x,y4),dim=1) #22
        x = self.C3_8(x) #23
        # Dectect 17 20 23 #24
        return x

    def forward(self, x):
        layers = []
        y = [] #save output
        x = layers[0](x) #0
        x = layers[1](x) #1
        x = layers[2](x) #2
        x = layers[3](x) #3
        x = layers[4](x) #4,y[0]
        y.append(x)
        x = layers[5](x) #5
        x = layers[6](x) #6,y[1]
        y.append(x)
        x = layers[7](x) #7
        x = layers[8](x) #8?
        x = layers[9](x) #9?
        x = layers[10](x) #10,y[2]
        y.append(x)
        x = layers[11](x) #11
        x = torch.cat((x,y[1]),dim=1) #12
        x = layers[13](x) #13
        x = layers[14](x) #14,y[3]
        y.append(x)
        x = layers[15](x) #15
        x = torch.cat((x,y[0]),dim=1) #16
        x = layers[17](x) #17
        x = layers[18](x) #18
        x = torch.cat((x,y[3]),dim=1) #19
        x = layers[20](x) #20
        x = layers[21](x) #21
        x = torch.cat((x,y[2]),dim=1) #22
        x = layers[23](x) #23
        # Dectect 17 20 23 #24
        return x

    def forward(self, x):
        layers = []
        y = [] #save output
        x = layers[0](x) #0
        x = layers[1](x) #1
        x = layers[2](x) #2
        x = layers[3](x) #3
        x = layers[4](x) #4,y[0]
        y.append(x)
        x = layers[5](x) #5
        x = layers[6](x) #6,y[1]
        y.append(x)
        x = layers[7](x) #7
        x = layers[8](x) #8?
        x = layers[9](x) #9?
        x = layers[10](x) #10,y[2]
        y.append(x)
        x = layers[11](x) #11
        x = layers[12](x) #12 x=cat(x,y[1])
        x = layers[13](x) #13
        x = layers[14](x) #14,y[3]
        y.append(x)
        x = layers[15](x) #15
        x = layers[16](x) #16 x=cat(x,y[0])
        x = layers[17](x) #17,y[4]
        y.append(x)
        x = layers[18](x) #18
        x = layers[19](x) #19 x=cat(x,y[3])
        x = layers[20](x) #20,y[5]
        y.append(x)
        x = layers[21](x) #21
        x = layers[22](x) #22 x=cat(x,y[2])
        x = layers[23](x) #23,y[6]
        y.append(x)
        x = layers[24](y[4:6]) #24 
        return x

save = []
for x in ([f] if isinstance(f, int) else f) :
    if x != -1 :
        save.extend(x) #接上一个list还是一个int

f = [] # from
save = []
for x in f :
    if x != -1 :
        save.extend(x)       