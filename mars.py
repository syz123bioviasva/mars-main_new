import sys
from engine.engine import MarsEngine


if __name__ == "__main__":
    mode = "pipe"
    nobuf = False
    cfgname = "c1.nano.full"  # 默认配置

    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-nobuf":
            nobuf = True
        elif arg == "-train":
            mode = "train"
        elif arg == "-eval":
            mode = "eval"
        elif arg == "-pipe":
            mode = "pipe"
        elif arg == "-cfgname" and i + 1 < len(sys.argv):
            cfgname = sys.argv[i + 1]

    MarsEngine(
        mode=mode,
        cfgname=cfgname,
        root="marsdata", # 注意项目运行root不要放在代码路径下
        nobuf=nobuf,
    ).run()
