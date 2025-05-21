from misc.misc import getModule


class MarsConfigFactory(object):
    @staticmethod
    def loadConfig(cfgname, mode, root, nobuf):
        module, baseName, tags = getModule("cfgops", cfgname)
        mcfg = module.mcfg(tags)
        mcfg.mode = mode
        mcfg.root = root
        mcfg.cfgname = cfgname
        mcfg.nobuf = nobuf
        return mcfg.finalize(tags)
