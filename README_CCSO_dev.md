### 构建
```
cd Build/linux
./build.sh <release | debug>

./build.sh <release | debug> --enable-lto   # for LTO build run
```
可执行二进制文件会生成在`Bin/Release` 和 `Bin/Debug`下面。

### CCSO
CCSO开关位于`EbCcso.h`中，使用宏`#define CCSO 1/0`控制

CCSO参数search以及滤波过程在`EbCdefProcess.c`的`svt_aom_cdef_kernel`中。

CCSO相关函数的声明和定义位于`EbCcso.h, EbCcso.c, EbPickccso.h, EbPickccso.c`;

编码端码流写入位于`EbEntropyCoding.c`，其中：
> write_sequence_header：序列级参数；   
> encode_ccso：帧级参数；   
> write_ccso：块级参数；

解码端在`read_sequence_header_obu`中读取序列级开关；`read_frame_ccso_params`和`EbDecParseFrame.c`的`parse_tile`中读取每一帧参数；`EbDecParseBlock.c`中的`read_ccso`读取每个块的参数；最后在`read_tile_group_obu`之中完成解码端滤波。

### 测试说明
目前CCSO是加在`CDEF_kernel`上，只需要在并行度初始化时把`load_default_buffer_configuration_settings`中的`cdef_process_init_count`设置成1就可以多线程运行（其他工具多线程，ccso过程单线程），极大提高运行速度。但是目前SVT-AV1其他编码工具比如DLF、CDEF等都进行了基于segment的并行处理优化，所以enctime_ratio并不适合在此类编码器上衡量编码工具的复杂度。

推荐测试命令行：
```
{enc}   -i {src_yuv} --lp 1 --frames 130 --width {w} --height {h} --tune 2 --preset 3 --fps 50 --rc 0 --qp 23 --enable-qm 1 --keyint 256 --hierarchical-levels 5 --enable-stat-report 1 --stat-file {log} -b {bin} -o {rec_yuv} 2>{terminal_log}

{av1dec_ccso} -i {bin} -threads 1 -parallel-frames 1 -w {width} -h {height} -bit-depth 8 -colour-space 420 -fps-frm 1 -fps-summary 1 -o {dec_yuv} 2>{term_log}
```