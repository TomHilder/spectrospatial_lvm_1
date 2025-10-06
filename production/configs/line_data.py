# line_data.py
# Just the line centres in Angstroms


LINE_CENTRES = {
    "HIλ3687": 3686.83,
    "HIλ3692": 3691.56,
    "HIλ3697": 3697.15,
    "HIλ3704": 3703.85,
    "HIλ3712": 3711.97,
    "[OII]λ3726": 3726.03,  # risk of blending
    "[OII]λ3729": 3728.82,  # risk of blending
    "HIλ3734": 3734.37,
    "HIλ3750": 3750.15,
    "[FeVII]λ3759": 3758.9,
    "HIλ3771": 3770.63,
    "HIλ3798": 3797.9,
    "HeIλ3820": 3819.61,
    "HIλ3835": 3835.38,
    "[NeIII]λ3869": 3868.75,
    "HeIλ3889": 3888.65,  # risk of blending
    "HIλ3889": 3889.05,  # risk of blending
    "CaIIλ3934": 3933.66,
    "HeIλ3965": 3964.73,  # risk of blending
    "[NeIII]λ3967": 3967.46,  # risk of blending
    "CaIIλ3968": 3968.47,  # risk of blending
    "Hϵλ3970": 3970.07,  # risk of blending
    "HeIλ4026": 4026.19,
    "[SII]λ4069": 4068.6,
    "[SII]λ4076": 4076.35,
    "Hδλ4102": 4101.77,
    "HeIλ4121": 4120.81,
    "[FeII]λ4177": 4177.21,
    "[FeV]λ4227": 4227.2,
    "[FeII]λ4244": 4243.98,
    "CIIλ4267": 4267.0,
    "[FeII]λ4287": 4287.4,
    "Hγλ4340": 4340.49,
    "[FeII]λ4358_1": 4358.1,  # risk of blending (NOTE MISSING???)
    "[FeII]λ4358_4": 4358.37,  # risk of blending
    "[FeII]λ4359": 4359.34,  # risk of blending
    "[OIII]λ4363": 4363.21,  # risk of blending
    "[FeII]λ4413_8": 4413.78,  # risk of blending (NOTE MISSING???)
    "[FeII]λ4414_5": 4414.45,  # risk of blending
    "[FeII]λ4416": 4416.27,  # risk of blending
    "[FeII]λ4452": 4452.11,
    "[FeII]λ4458": 4457.95,
    "[FeII]λ4470": 4470.29,  # risk of blending
    "HeIλ4471": 4471.48,  # risk of blending
    "[FeII]λ4475": 4474.91,  # risk of blending
    "[NiII]λ4485": 4485.21,
    "[MgI]λ4562": 4562.48,
    "[MgI]λ4571": 4571.1,
    "[FeII]λ4632": 4632.27,
    "[FeIII]λ4658": 4658.1,
    "HeIIλ4686": 4685.68,
    "[FeIII]λ4702": 4701.62,
    "[ArIV]λ4711": 4711.33,  # risk of blending
    "HeIλ4713": 4713.14,  # risk of blending
    "[NeIV]λ4724": 4724.17,
    "[FeIII]λ4734": 4733.93,
    "[ArIV]λ4740": 4740.2,
    "[FeIII]λ4755": 4754.83,
    "[FeIII]λ4770": 4769.6,
    "[FeII]λ4775": 4774.74,  # risk of blending
    "[FeIII]λ4778": 4777.88,  # risk of blending
    "[FeIII]λ4814": 4813.9,  # risk of blending
    "[FeII]λ4815": 4814.55,  # risk of blending
    "Hβλ4861": 4861.36,
    "[FeIII]λ4881": 4881.11,
    "[FeII]λ4890": 4889.63,  # risk of blending
    "[FeVII]λ4893": 4893.4,  # risk of blending
    "[FeII]λ4905": 4905.35,
    "HeIλ4922": 4921.93,  # risk of blending
    "[FeIII]λ4924": 4924.5,  # risk of blending
    "[FeIII]λ4930": 4930.5,
    "[FeVII]λ4943": 4942.5,
    "[OIII]λ4959": 4958.91,
    "[FeVI]λ4972_5": 4972.5,  # risk of blending
    "[FeII]λ4973_3": 4973.39,  # risk of blending
    "[FeIII]λ4986": 4985.9,
    "[OIII]λ5007": 5006.84,
    "HeIλ5016": 5015.68,
    "[FeII]λ5039": 5039.1,
    "[FeII]λ5072": 5072.4,
    "[FeII]λ5108": 5107.95,  # risk of blending
    "[FeII]λ5112": 5111.63,  # risk of blending
    "[FeVI]λ5146": 5145.8,
    "[FeII]λ5158": 5158.0,  # risk of blending
    "[FeVII]λ5159": 5158.9,  # risk of blending
    "[FeVI]λ5176": 5176.0,
    "[FeII]λ5185": 5184.8,
    "[ArIII]λ5192": 5191.82,
    "[NI]λ5198": 5197.9,  # risk of blending
    "[NI]λ5200": 5200.26,  # risk of blending
    "[FeII]λ5220": 5220.06,
    "[FeII]λ5262": 5261.61,
    "[FeII]λ5269": 5268.88,  # risk of blending
    "[FeIII]λ5270": 5270.3,  # risk of blending
    "[FeII]λ5273": 5273.38,  # risk of blending
    "[FeVI]λ5278": 5277.8,
    "[FeII]λ5297": 5296.84,
    "[FeXIV]λ5303": 5302.86,
    "[CaV]λ5309": 5309.18,
    "[FeII]λ5334": 5333.65,  # risk of blending
    "[FeVI]λ5335": 5335.2,  # risk of blending
    "[FeII]λ5376": 5376.47,
    "HeIIλ5412": 5411.52,  # risk of blending
    "[FeII]λ5413": 5412.64,  # risk of blending
    "[FeVI]λ5424": 5424.2,  # risk of blending
    "[FeVI]λ5427": 5426.6,  # risk of blending
    "[FeVI]λ5485": 5484.8,
    "[ClIII]λ5518": 5517.71,
    "[FeII]λ5527": 5527.33,
    "[OI]λ5577": 5577.34,
    "[FeVI]λ5631": 5631.1,
    "NIIλ5680": 5679.56,
    "[FeVI]λ5677": 5677.0,
    "[FeVII]λ5721": 5720.7,
    "[NII]λ5755": 5754.59,
    "HeIλ5876": 5876.0,
    "NaIλ5890": 5889.95,
    "NaIλ5896": 5895.92,
    "[FeVII]λ6087": 6087.0,
    "[OI]λ6300": 6300.3,
    "[SIII]λ6312": 6312.06,
    "[OI]λ6364": 6363.78,
    "[FeX]λ6375": 6374.51,
    "[ArV]λ6435": 6435.1,
    "[NII]λ6548": 6548.05,
    "Hαλ6563": 6562.85,
    "CIIλ6578": 6578.05,
    "[NII]λ6583": 6583.45,
    "HeIλ6678": 6678.15,
    "[SII]λ6716": 6716.44,
    "[SII]λ6731": 6730.82,
    "FeIλ6855": 6855.18,
    "[ArV]λ7006": 7005.67,
    "HeIλ7065": 7065.19,
    "[ArIII]λ7136": 7135.8,
    "[FeII]λ7155": 7155.14,
    "[FeII]λ7172": 7171.98,
    "CIIλ7236": 7236.0,
    "HeIλ7281": 7281.35,
    "[FeI]λ7290": 7290.42,  # risk of blending
    "[CaII]λ7291": 7291.46,  # risk of blending
    "[OII]λ7319": 7318.92,
    "[CaII]λ7324": 7323.88,
    "[OII]λ7330": 7329.66,
    "[NiII]λ7378": 7377.83,
    "[FeII]λ7388": 7388.16,
    "[NiII]λ7412": 7411.61,
    "[FeII]λ7453": 7452.5,
    "[FeII]λ7638": 7637.52,
    "[FeII]λ7686": 7686.19,  # risk of blending
    "[FeII]λ7687": 7686.9,  # risk of blending
    "[ArIII]λ7751": 7751.06,
    "OIλ7774": 7774.0,
    "[FeXI]λ7892": 7891.8,
    "[CrII]λ8000": 7999.85,
    "[CrII]λ8125": 8125.22,
    "[CrII]λ8230": 8229.55,
    "HeIIλ8237": 8236.77,  # risk of blending
    "HIλ8239": 8239.24,  # risk of blending
    "HIλ8241": 8240.82,  # risk of blending
    "HIλ8243": 8242.51,  # risk of blending
    "HIλ8244": 8244.32,  # risk of blending
    "HIλ8246": 8246.27,  # risk of blending
    "HIλ8248": 8248.35,  # risk of blending
    "HIλ8251": 8250.6,  # risk of blending
    "HIλ8253": 8253.03,  # risk of blending
    "HIλ8256": 8255.65,  # risk of blending
    "HIλ8258": 8258.49,  # risk of blending
    "HIλ8262": 8261.57,  # risk of blending
    "HIλ8265": 8264.92,  # risk of blending
    "HIλ8269": 8268.57,  # risk of blending
    "HIλ8273": 8272.56,  # risk of blending
    "HIλ8277": 8276.94,
    "HIλ8282": 8281.75,
    "HIλ8287": 8287.06,
    "HIλ8293": 8292.94,
    "HIλ8299": 8299.47,  # risk of blending
    "[NiII]λ8311": 8300.99,  # risk of blending
    "HIλ8307": 8306.75,  # risk of blending
    "[CrII]λ8308": 8308.39,  # risk of blending
    "HIλ8315": 8314.89,
    "HIλ8324": 8324.06,
    "HIλ8334": 8334.42,
    "HIλ8346": 8345.55,
    "[CrII]λ8358": 8357.51,  # risk of blending
    "HIλ8359": 8359.0,  # risk of blending
    "HIλ8374": 8374.48,
    "HIλ8392": 8392.4,
    "OIλ8446": 8446.0,
    "HIλ8467": 8467.25,
    "CaIIλ8498": 8498.02,
    "HIλ8502": 8502.48,
    "CaIIλ8542": 8542.09,  # risk of blending
    "HIλ8545": 8545.38,  # risk of blending
    "[ClII]λ8579": 8578.7,
    "HIλ8598": 8598.39,
    "[FeII]λ8617": 8616.96,
    "CaIIλ8662": 8662.14,  # risk of blending
    "HIλ8665": 8665.02,  # risk of blending
    "[CI]λ8727": 8727.13,
    "HIλ8750": 8750.47,
    "HIλ8863": 8862.78,
    "[FeII]λ8892": 8891.88,
    "HIλ9015": 9014.91,
    "[FeII]λ9033": 9033.45,
    "[FeII]λ9052": 9051.92,
    "[SIII]λ9069": 9069.0,
    "[ClII]λ9124": 9123.6,
    "[FeII]λ9227": 9226.6,  # risk of blending
    "HIλ9229": 9229.02,  # risk of blending
    "OIλ9266": 9266.0,  # risk of blending
    "[FeII]λ9268": 9267.54,  # risk of blending
    "[FeII]λ9399": 9399.02,
    "[FeII]λ9471": 9470.93,
    "[SIII]λ9531": 9531.1,
    "HIλ9546": 9545.97,
    "[FeII]λ9682": 9682.13,
}

OLD_LINE_CENTRES = {
    "[OII]λ3727": 3727,  # This one is a doublet so idk what to do yet
    "Hδλ4102": 4102,
    "Hγλ4340": 4340,
    "CIIλ4267": 4267,  # (RL) Nothing in stacked spectrum
    "[OIII]λ4363": 4363,  # Auroral line Nothing in stacked spectrum
    "HeIλ4471": 4471,
    "OIIλ4650": 4650,  # (RL; V1 multiplet) Stacked spectrum tantalising (NOTE MISSING FROM NEW LIST)
    "OIIλ4660": 4660,  # (RL; V1 multiplet) Ditto (NOTE MISSING FROM NEW LIST)
    "Hβλ4861": 4861,
    "[OIII]λ5007": 5007,
    "NIIλ5680": 5680,  # (RL) Nothing in stacked spectrum
    "[NII]λ5755": 5755,
    "[SIII]λ6312": 6312,
    "[NII]λ6548": 6548,
    "Hαλ6563": 6563,
    "CIIλ6578": 6578,  # (RL) This one is the most promising
    "[NII]λ6584": 6584,  # NOTE this one changed name/key from [NII]λ6584 to [NII]λ6583
    "HeIλ6678": 6678,
    "[SII]λ6716": 6716,
    "[SII]λ6731": 6731,
    "[OII]λ7320": 7320,  # NOTE this one changed name/key from [OII]λ7320 to [OII]λ7319
    "OIλ7774": 7774,  # (Fluoresecent) Very strong in stacked spectrum (I think this one is the sky lmao)
    "[SIII]λ9069": 9069,
    "[SIII]λ9531": 9531,
}
