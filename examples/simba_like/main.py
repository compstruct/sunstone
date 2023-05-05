import sys, os
sys.path.append(os.getcwd())
from examples.simba_like.cnn_optimizer_wrapper import *
from commons.software.cnns import ResNet18
from cnn_optimizer_wrapper import CNNOptimizer

# ****** Simba architecture configurations ******

# ****** Memory hierarchy of Simba except
#        MAC Regs that are hardcoded in the
#        optimizer (lines 52-60 of cnn_optimizer_wrapper) ******
memrs_flat = {
    "reg": 1,
    "L1_ifmap": 8 * 1024,
    "L1_w": 32 * 1024,
    "L1_ofmap": 1024,
    "L2": 64 * 1024,
}
# ******* this is the memory hierarchy format that our wrapper accepts 
#         (look into cnn_optimizer_wrapper)                            *******
smba_memrs = {
    "reg" : memrs_flat["reg"],
    "L1" : (memrs_flat["L1_ifmap"], memrs_flat["L1_w"], memrs_flat["L1_ofmap"]),
    "L2" : memrs_flat["L2"]
}

# ****** Access energies that can come from Accelergy 
#        (CACTI + ALADDIN) or any other source        ******
smba_acs_enrgs = {
    # Register inside vector MAC
    "reg_ifmap": 0,
    "reg_w": 0.49,
    "reg_ofmap": 0,
    # PE SRAM
    "L1_ifmap": 1.24,
    "L1_w": 5.59,
    "L1_ofmap": 1.01,
    "L1_NoC": 0.1,
    # Global SRAM
    "L2_ifmap": 5.47,
    "L2_w": 5.5,
    "L2_ofmap": 5.71,
    # Global SRAM
    "DR_ifmap": 64,
    "DR_w": 64,
    "DR_ofmap": 64,
}

# ******* dimensions of the vector MAC *******
smba_pe_sptl = {"X": 8, "Y": 8}

# ******* dimensions of the PE array *******
smba_sptl = {"X": 4, "Y": 4}

smba_bws = [None, (64, 64), (256, 256), None]

# ******* this list indicate whether each operand has 
#         a dedicated buffer at each level & will be 
#         used to optimize for uneven mapping which Timeloop
#         also supports. If uneven mapping will not be used, 
#         set this to None                                   *******
sprt_strct = [(False, True, False), (True, True, True), (False, False, False)]

# ******* the tensor to keep inside vector MAC registers 
#         here we by-pass both IFMAP and OFMAP and only keep weigts
smba_bypass = {  
                        "reg": (True, False, True), 
                        "L1": (False, False, False),
                        "L2" : (False, True, False)}

cnn = ResNet18(n=16).get_layers()
cnn_non_duplicate = [l for l in cnn if not l.duplicate]
lyr1 = cnn_non_duplicate[0]
lyr = {
            "N": lyr1.n,
            "C": lyr1.c,
            "K": lyr1.m,
            "R": lyr1.r,
            "S": lyr1.s,
            "P": lyr1.p,
            "Q": lyr1.q,
        }

opt = CNNOptimizer(
            lyr, 
            smba_memrs, 
            smba_acs_enrgs, 
            smba_bws, 
            smba_sptl, 
            smba_pe_sptl, 
            smba_bypass, 
            sprt_strct, 
            memrs_flat
        )
tls, unevn_mp, ords, cost  = opt.solve(threads=8, optimize_uneven_mapping=False)
print(opt.optimal_mapping)
