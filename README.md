
NPDface人脸检测程序包含训练和检测
train文件夹内为训练程序，需要使用cmake构建工程，支持gpu（cuda）加速
detect文件问检测程序，c++编写，同样适用cmake管理工程，所有程序只依赖opencv进行图片读取
data文件包含三个可用模型包含测试曲线，使用fddb数据库测试得出。
