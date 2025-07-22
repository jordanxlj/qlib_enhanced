#!/usr/bin/env python3
"""
CSI300 数据性能测试脚本
专门用于测试qlib表达式在CSI300股票池上的性能
"""

import time
import psutil
import numpy as np
import pandas as pd
import yaml
import sys
import gc
import warnings
from typing import List, Dict, Tuple, Optional
import traceback

# 添加qlib路径
sys.path.insert(0, '/d:/code/stock/qlib')

try:
    import qlib
    from qlib.data import D
    from qlib.config import C
    
    # 初始化qlib
    qlib.init(provider_uri="data", region="cn")
    
except ImportError as e:
    print(f"❌ 无法导入qlib: {e}")
    sys.exit(1)

class CSI300PerformanceTester:
    """CSI300性能测试器"""
    
    def __init__(self):
        self.results = []
        
    def get_csi300_instruments(self, sample_size: int = None) -> List[str]:
        """获取CSI300成分股列表"""
        try:
            # 使用qlib获取CSI300成分股
            instruments = D.instruments(market="csi300")
            
            if sample_size and len(instruments) > sample_size:
                # 随机采样，但保证包含一些知名股票
                import random
                
                # 确保包含一些知名股票
                priority_stocks = [
                    "SH600000",
                    "SH600001",
                    "SH600002",
                    "SH600003",
                    "SH600004",
                    "SH600005",
                    "SH600006",
                    "SH600007",
                    "SH600008",
                    "SH600009",
                    "SH600010",
                    "SH600011",
                    "SH600012",
                    "SH600015",
                    "SH600016",
                    "SH600017",
                    "SH600018",
                    "SH600019",
                    "SH600020",
                    "SH600021",
                    "SH600022",
                    "SH600023",
                    "SH600025",
                    "SH600026",
                    "SH600027",
                    "SH600028",
                    "SH600029",
                    "SH600030",
                    "SH600031",
                    "SH600033",
                    "SH600035",
                    "SH600036",
                    "SH600037",
                    "SH600038",
                    "SH600039",
                    "SH600048",
                    "SH600050",
                    "SH600057",
                    "SH600058",
                    "SH600060",
                    "SH600061",
                    "SH600062",
                    "SH600066",
                    "SH600068",
                    "SH600072",
                    "SH600073",
                    "SH600074",
                    "SH600078",
                    "SH600079",
                    "SH600085",
                    "SH600087",
                    "SH600088",
                    "SH600089",
                    "SH600091",
                    "SH600096",
                    "SH600098",
                    "SH600100",
                    "SH600102",
                    "SH600103",
                    "SH600104",
                    "SH600108",
                    "SH600109",
                    "SH600110",
                    "SH600111",
                    "SH600115",
                    "SH600117",
                    "SH600118",
                    "SH600121",
                    "SH600123",
                    "SH600125",
                    "SH600126",
                    "SH600132",
                    "SH600135",
                    "SH600138",
                    "SH600143",
                    "SH600150",
                    "SH600151",
                    "SH600153",
                    "SH600157",
                    "SH600158",
                    "SH600160",
                    "SH600161",
                    "SH600166",
                    "SH600169",
                    "SH600170",
                    "SH600171",
                    "SH600176",
                    "SH600177",
                    "SH600183",
                    "SH600188",
                    "SH600190",
                    "SH600196",
                    "SH600198",
                    "SH600200",
                    "SH600205",
                    "SH600207",
                    "SH600208",
                    "SH600210",
                    "SH600215",
                    "SH600216",
                    "SH600219",
                    "SH600220",
                    "SH600221",
                    "SH600228",
                    "SH600231",
                    "SH600233",
                    "SH600236",
                    "SH600239",
                    "SH600246",
                    "SH600251",
                    "SH600252",
                    "SH600256",
                    "SH600259",
                    "SH600266",
                    "SH600267",
                    "SH600269",
                    "SH600270",
                    "SH600271",
                    "SH600276",
                    "SH600277",
                    "SH600282",
                    "SH600296",
                    "SH600297",
                    "SH600299",
                    "SH600300",
                    "SH600307",
                    "SH600308",
                    "SH600309",
                    "SH600312",
                    "SH600315",
                    "SH600316",
                    "SH600317",
                    "SH600320",
                    "SH600325",
                    "SH600331",
                    "SH600332",
                    "SH600333",
                    "SH600339",
                    "SH600340",
                    "SH600346",
                    "SH600348",
                    "SH600350",
                    "SH600352",
                    "SH600357",
                    "SH600361",
                    "SH600362",
                    "SH600369",
                    "SH600372",
                    "SH600373",
                    "SH600376",
                    "SH600377",
                    "SH600380",
                    "SH600383",
                    "SH600390",
                    "SH600395",
                    "SH600398",
                    "SH600399",
                    "SH600403",
                    "SH600406",
                    "SH600408",
                    "SH600410",
                    "SH600415",
                    "SH600418",
                    "SH600426",
                    "SH600428",
                    "SH600432",
                    "SH600436",
                    "SH600438",
                    "SH600446",
                    "SH600456",
                    "SH600460",
                    "SH600472",
                    "SH600481",
                    "SH600482",
                    "SH600485",
                    "SH600487",
                    "SH600489",
                    "SH600497",
                    "SH600498",
                    "SH600500",
                    "SH600501",
                    "SH600508",
                    "SH600515",
                    "SH600516",
                    "SH600517",
                    "SH600518",
                    "SH600519",
                    "SH600521",
                    "SH600522",
                    "SH600528",
                    "SH600535",
                    "SH600546",
                    "SH600547",
                    "SH600548",
                    "SH600549",
                    "SH600550",
                    "SH600566",
                    "SH600569",
                    "SH600570",
                    "SH600578",
                    "SH600581",
                    "SH600582",
                    "SH600583",
                    "SH600584",
                    "SH600585",
                    "SH600588",
                    "SH600591",
                    "SH600595",
                    "SH600596",
                    "SH600597",
                    "SH600598",
                    "SH600600",
                    "SH600601",
                    "SH600602",
                    "SH600606",
                    "SH600608",
                    "SH600611",
                    "SH600616",
                    "SH600621",
                    "SH600627",
                    "SH600628",
                    "SH600630",
                    "SH600631",
                    "SH600633",
                    "SH600635",
                    "SH600637",
                    "SH600638",
                    "SH600639",
                    "SH600641",
                    "SH600642",
                    "SH600643",
                    "SH600648",
                    "SH600649",
                    "SH600652",
                    "SH600653",
                    "SH600654",
                    "SH600655",
                    "SH600657",
                    "SH600660",
                    "SH600662",
                    "SH600663",
                    "SH600664",
                    "SH600666",
                    "SH600674",
                    "SH600675",
                    "SH600682",
                    "SH600685",
                    "SH600688",
                    "SH600690",
                    "SH600694",
                    "SH600703",
                    "SH600704",
                    "SH600705",
                    "SH600707",
                    "SH600710",
                    "SH600717",
                    "SH600718",
                    "SH600724",
                    "SH600726",
                    "SH600732",
                    "SH600733",
                    "SH600737",
                    "SH600739",
                    "SH600740",
                    "SH600741",
                    "SH600744",
                    "SH600745",
                    "SH600747",
                    "SH600748",
                    "SH600754",
                    "SH600757",
                    "SH600760",
                    "SH600761",
                    "SH600763",
                    "SH600770",
                    "SH600779",
                    "SH600780",
                    "SH600782",
                    "SH600783",
                    "SH600786",
                    "SH600787",
                    "SH600790",
                    "SH600795",
                    "SH600797",
                    "SH600803",
                    "SH600804",
                    "SH600805",
                    "SH600808",
                    "SH600809",
                    "SH600811",
                    "SH600812",
                    "SH600816",
                    "SH600820",
                    "SH600823",
                    "SH600827",
                    "SH600832",
                    "SH600834",
                    "SH600835",
                    "SH600837",
                    "SH600838",
                    "SH600839",
                    "SH600845",
                    "SH600848",
                    "SH600851",
                    "SH600854",
                    "SH600859",
                    "SH600863",
                    "SH600866",
                    "SH600867",
                    "SH600868",
                    "SH600871",
                    "SH600872",
                    "SH600873",
                    "SH600874",
                    "SH600875",
                    "SH600879",
                    "SH600880",
                    "SH600881",
                    "SH600884",
                    "SH600886",
                    "SH600887",
                    "SH600893",
                    "SH600894",
                    "SH600895",
                    "SH600900",
                    "SH600905",
                    "SH600909",
                    "SH600918",
                    "SH600919",
                    "SH600926",
                    "SH600928",
                    "SH600938",
                    "SH600941",
                    "SH600958",
                    "SH600959",
                    "SH600961",
                    "SH600968",
                    "SH600970",
                    "SH600971",
                    "SH600977",
                    "SH600978",
                    "SH600989",
                    "SH600997",
                    "SH600998",
                    "SH600999",
                    "SH601001",
                    "SH601002",
                    "SH601003",
                    "SH601005",
                    "SH601006",
                    "SH601009",
                    "SH601012",
                    "SH601016",
                    "SH601018",
                    "SH601021",
                    "SH601058",
                    "SH601059",
                    "SH601066",
                    "SH601077",
                    "SH601088",
                    "SH601098",
                    "SH601099",
                    "SH601100",
                    "SH601101",
                    "SH601106",
                    "SH601107",
                    "SH601108",
                    "SH601111",
                    "SH601117",
                    "SH601118",
                    "SH601127",
                    "SH601136",
                    "SH601138",
                    "SH601139",
                    "SH601155",
                    "SH601158",
                    "SH601162",
                    "SH601163",
                    "SH601166",
                    "SH601168",
                    "SH601169",
                    "SH601179",
                    "SH601186",
                    "SH601198",
                    "SH601211",
                    "SH601212",
                    "SH601216",
                    "SH601225",
                    "SH601228",
                    "SH601229",
                    "SH601231",
                    "SH601233",
                    "SH601236",
                    "SH601238",
                    "SH601258",
                    "SH601268",
                    "SH601288",
                    "SH601298",
                    "SH601299",
                    "SH601318",
                    "SH601319",
                    "SH601328",
                    "SH601333",
                    "SH601336",
                    "SH601360",
                    "SH601369",
                    "SH601375",
                    "SH601377",
                    "SH601390",
                    "SH601398",
                    "SH601519",
                    "SH601555",
                    "SH601558",
                    "SH601566",
                    "SH601577",
                    "SH601588",
                    "SH601600",
                    "SH601601",
                    "SH601607",
                    "SH601608",
                    "SH601611",
                    "SH601615",
                    "SH601618",
                    "SH601628",
                    "SH601633",
                    "SH601658",
                    "SH601666",
                    "SH601668",
                    "SH601669",
                    "SH601688",
                    "SH601689",
                    "SH601696",
                    "SH601698",
                    "SH601699",
                    "SH601717",
                    "SH601718",
                    "SH601727",
                    "SH601728",
                    "SH601766",
                    "SH601788",
                    "SH601799",
                    "SH601800",
                    "SH601801",
                    "SH601808",
                    "SH601816",
                    "SH601818",
                    "SH601825",
                    "SH601828",
                    "SH601838",
                    "SH601857",
                    "SH601865",
                    "SH601866",
                    "SH601868",
                    "SH601872",
                    "SH601877",
                    "SH601878",
                    "SH601881",
                    "SH601888",
                    "SH601898",
                    "SH601899",
                    "SH601901",
                    "SH601916",
                    "SH601918",
                    "SH601919",
                    "SH601928",
                    "SH601929",
                    "SH601933",
                    "SH601939",
                    "SH601958",
                    "SH601966",
                    "SH601969",
                    "SH601985",
                    "SH601988",
                    "SH601989",
                    "SH601990",
                    "SH601991",
                    "SH601992",
                    "SH601995",
                    "SH601997",
                    "SH601998",
                    "SH603000",
                    "SH603019",
                    "SH603087",
                    "SH603156",
                    "SH603160",
                    "SH603185",
                    "SH603195",
                    "SH603233",
                    "SH603259",
                    "SH603260",
                    "SH603288",
                    "SH603290",
                    "SH603296",
                    "SH603338",
                    "SH603369",
                    "SH603392",
                    "SH603486",
                    "SH603501",
                    "SH603517",
                    "SH603658",
                    "SH603659",
                    "SH603699",
                    "SH603799",
                    "SH603806",
                    "SH603833",
                    "SH603858",
                    "SH603882",
                    "SH603885",
                    "SH603899",
                    "SH603939",
                    "SH603986",
                    "SH603993",
                    "SH605117",
                    "SH605499",
                    "SH688005",
                    "SH688008",
                    "SH688009",
                    "SH688012",
                    "SH688036",
                    "SH688041",
                    "SH688047",
                    "SH688065",
                    "SH688082",
                    "SH688111",
                    "SH688126",
                    "SH688169",
                    "SH688187",
                    "SH688223",
                    "SH688256",
                    "SH688271",
                    "SH688303",
                    "SH688363",
                    "SH688396",
                    "SH688472",
                    "SH688506",
                    "SH688561",
                    "SH688599",
                    "SH688981",
                    "SHT00018",
                    "SZ000001",
                    "SZ000002",
                    "SZ000008",
                    "SZ000009",
                    "SZ000012",
                    "SZ000016",
                    "SZ000021",
                    "SZ000022",
                    "SZ000024",
                    "SZ000027",
                    "SZ000029",
                    "SZ000031",
                    "SZ000036",
                    "SZ000039",
                    "SZ000046",
                    "SZ000059",
                    "SZ000060",
                    "SZ000061",
                    "SZ000063",
                    "SZ000066",
                    "SZ000068",
                    "SZ000069",
                    "SZ000088",
                    "SZ000089",
                    "SZ000096",
                    "SZ000099",
                    "SZ000100",
                    "SZ000156",
                    "SZ000157",
                    "SZ000166",
                    "SZ000301",
                    "SZ000333",
                    "SZ000338",
                    "SZ000400",
                    "SZ000401",
                    "SZ000402",
                    "SZ000406",
                    "SZ000408",
                    "SZ000410",
                    "SZ000413",
                    "SZ000415",
                    "SZ000420",
                    "SZ000422",
                    "SZ000423",
                    "SZ000425",
                    "SZ000429",
                    "SZ000488",
                    "SZ000498",
                    "SZ000503",
                    "SZ000507",
                    "SZ000511",
                    "SZ000518",
                    "SZ000520",
                    "SZ000527",
                    "SZ000528",
                    "SZ000532",
                    "SZ000533",
                    "SZ000536",
                    "SZ000538",
                    "SZ000539",
                    "SZ000540",
                    "SZ000541",
                    "SZ000543",
                    "SZ000550",
                    "SZ000553",
                    "SZ000555",
                    "SZ000559",
                    "SZ000562",
                    "SZ000568",
                    "SZ000571",
                    "SZ000572",
                    "SZ000573",
                    "SZ000581",
                    "SZ000596",
                    "SZ000598",
                    "SZ000599",
                    "SZ000601",
                    "SZ000607",
                    "SZ000612",
                    "SZ000617",
                    "SZ000618",
                    "SZ000623",
                    "SZ000625",
                    "SZ000627",
                    "SZ000629",
                    "SZ000630",
                    "SZ000631",
                    "SZ000636",
                    "SZ000651",
                    "SZ000652",
                    "SZ000656",
                    "SZ000659",
                    "SZ000661",
                    "SZ000666",
                    "SZ000667",
                    "SZ000671",
                    "SZ000680",
                    "SZ000682",
                    "SZ000685",
                    "SZ000686",
                    "SZ000690",
                    "SZ000698",
                    "SZ000703",
                    "SZ000707",
                    "SZ000708",
                    "SZ000709",
                    "SZ000712",
                    "SZ000717",
                    "SZ000718",
                    "SZ000723",
                    "SZ000725",
                    "SZ000726",
                    "SZ000727",
                    "SZ000728",
                    "SZ000729",
                    "SZ000733",
                    "SZ000735",
                    "SZ000737",
                    "SZ000738",
                    "SZ000750",
                    "SZ000751",
                    "SZ000755",
                    "SZ000758",
                    "SZ000761",
                    "SZ000763",
                    "SZ000767",
                    "SZ000768",
                    "SZ000776",
                    "SZ000778",
                    "SZ000780",
                    "SZ000783",
                    "SZ000786",
                    "SZ000792",
                    "SZ000793",
                    "SZ000800",
                    "SZ000806",
                    "SZ000807",
                    "SZ000817",
                    "SZ000822",
                    "SZ000823",
                    "SZ000825",
                    "SZ000826",
                    "SZ000828",
                    "SZ000829",
                    "SZ000831",
                    "SZ000839",
                    "SZ000858",
                    "SZ000860",
                    "SZ000866",
                    "SZ000869",
                    "SZ000875",
                    "SZ000876",
                    "SZ000877",
                    "SZ000878",
                    "SZ000883",
                    "SZ000886",
                    "SZ000895",
                    "SZ000897",
                    "SZ000898",
                    "SZ000900",
                    "SZ000912",
                    "SZ000916",
                    "SZ000917",
                    "SZ000920",
                    "SZ000921",
                    "SZ000927",
                    "SZ000930",
                    "SZ000932",
                    "SZ000933",
                    "SZ000937",
                    "SZ000938",
                    "SZ000939",
                    "SZ000949",
                    "SZ000951",
                    "SZ000956",
                    "SZ000959",
                    "SZ000960",
                    "SZ000961",
                    "SZ000962",
                    "SZ000963",
                    "SZ000968",
                    "SZ000969",
                    "SZ000970",
                    "SZ000975",
                    "SZ000977",
                    "SZ000983",
                    "SZ000997",
                    "SZ000999",
                    "SZ001289",
                    "SZ001391",
                    "SZ001965",
                    "SZ001979",
                    "SZ002001",
                    "SZ002007",
                    "SZ002008",
                    "SZ002010",
                    "SZ002024",
                    "SZ002025",
                    "SZ002027",
                    "SZ002028",
                    "SZ002032",
                    "SZ002038",
                    "SZ002044",
                    "SZ002049",
                    "SZ002050",
                    "SZ002051",
                    "SZ002052",
                    "SZ002064",
                    "SZ002065",
                    "SZ002069",
                    "SZ002073",
                    "SZ002074",
                    "SZ002078",
                    "SZ002081",
                    "SZ002083",
                    "SZ002085",
                    "SZ002092",
                    "SZ002097",
                    "SZ002106",
                    "SZ002110",
                    "SZ002120",
                    "SZ002122",
                    "SZ002128",
                    "SZ002129",
                    "SZ002131",
                    "SZ002142",
                    "SZ002146",
                    "SZ002152",
                    "SZ002153",
                    "SZ002155",
                    "SZ002157",
                    "SZ002174",
                    "SZ002179",
                    "SZ002180",
                    "SZ002183",
                    "SZ002194",
                    "SZ002195",
                    "SZ002202",
                    "SZ002230",
                    "SZ002236",
                    "SZ002241",
                    "SZ002242",
                    "SZ002244",
                    "SZ002252",
                    "SZ002269",
                    "SZ002271",
                    "SZ002275",
                    "SZ002292",
                    "SZ002294",
                    "SZ002299",
                    "SZ002304",
                    "SZ002310",
                    "SZ002311",
                    "SZ002344",
                    "SZ002352",
                    "SZ002353",
                    "SZ002371",
                    "SZ002375",
                    "SZ002378",
                    "SZ002384",
                    "SZ002385",
                    "SZ002399",
                    "SZ002400",
                    "SZ002405",
                    "SZ002410",
                    "SZ002411",
                    "SZ002414",
                    "SZ002415",
                    "SZ002416",
                    "SZ002422",
                    "SZ002424",
                    "SZ002426",
                    "SZ002429",
                    "SZ002431",
                    "SZ002450",
                    "SZ002456",
                    "SZ002459",
                    "SZ002460",
                    "SZ002463",
                    "SZ002465",
                    "SZ002466",
                    "SZ002468",
                    "SZ002470",
                    "SZ002475",
                    "SZ002493",
                    "SZ002498",
                    "SZ002500",
                    "SZ002508",
                    "SZ002555",
                    "SZ002558",
                    "SZ002568",
                    "SZ002570",
                    "SZ002572",
                    "SZ002594",
                    "SZ002600",
                    "SZ002601",
                    "SZ002602",
                    "SZ002603",
                    "SZ002607",
                    "SZ002608",
                    "SZ002624",
                    "SZ002625",
                    "SZ002648",
                    "SZ002653",
                    "SZ002673",
                    "SZ002709",
                    "SZ002714",
                    "SZ002736",
                    "SZ002739",
                    "SZ002756",
                    "SZ002773",
                    "SZ002791",
                    "SZ002797",
                    "SZ002812",
                    "SZ002821",
                    "SZ002831",
                    "SZ002839",
                    "SZ002841",
                    "SZ002916",
                    "SZ002920",
                    "SZ002925",
                    "SZ002938",
                    "SZ002939",
                    "SZ002945",
                    "SZ002958",
                    "SZ003816",
                    "SZ300002",
                    "SZ300003",
                    "SZ300014",
                    "SZ300015",
                    "SZ300017",
                    "SZ300024",
                    "SZ300027",
                    "SZ300033",
                    "SZ300058",
                    "SZ300059",
                    "SZ300070",
                    "SZ300072",
                    "SZ300085",
                    "SZ300104",
                    "SZ300122",
                    "SZ300124",
                    "SZ300133",
                    "SZ300136",
                    "SZ300142",
                    "SZ300144",
                    "SZ300146",
                    "SZ300168",
                    "SZ300182",
                    "SZ300207",
                    "SZ300223",
                    "SZ300251",
                    "SZ300274",
                    "SZ300296",
                    "SZ300308",
                    "SZ300315",
                    "SZ300316",
                    "SZ300347",
                    "SZ300394",
                    "SZ300408",
                    "SZ300413",
                    "SZ300418",
                    "SZ300433",
                    "SZ300442",
                    "SZ300450",
                    "SZ300454",
                    "SZ300496",
                    "SZ300498",
                    "SZ300502",
                    "SZ300529",
                    "SZ300558",
                    "SZ300595",
                    "SZ300601",
                    "SZ300628",
                    "SZ300661",
                    "SZ300676",
                    "SZ300677",
                    "SZ300750",
                    "SZ300751",
                    "SZ300759",
                    "SZ300760",
                    "SZ300763",
                    "SZ300769",
                    "SZ300782",
                    "SZ300832",
                    "SZ300866",
                    "SZ300888",
                    "SZ300896",
                    "SZ300919",
                    "SZ300957",
                    "SZ300979",
                    "SZ300999",
                    "SZ301236",
                    "SZ301269",
                    "SZ302132",
                ]
                
                available_priority = [s for s in priority_stocks if s in instruments]
                remaining_slots = max(0, sample_size - len(available_priority))
                
                other_stocks = [s for s in instruments if s not in available_priority]
                random.shuffle(other_stocks)
                
                selected = available_priority + other_stocks[:remaining_slots]
                return selected[:sample_size]
            
            return instruments
            
        except Exception as e:
            print(f"⚠️  无法获取CSI300成分股，使用默认股票池: {e}")
            # 备用股票池
            return [
                "SH600519", "SZ000858", "SH600036", "SZ000001", "SH601318",
                "SH600276", "SZ000002", "SH601166", "SH600000", "SZ002415"
            ]
    
    def measure_memory_usage(self) -> float:
        """测量当前内存使用量 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def test_basic_features(self, 
                          instruments: List[str],
                          start_time: str = "2020-01-01",
                          end_time: str = "2020-03-01") -> List[Dict]:
        """测试基础特征性能"""
        
        basic_features = [
            ("$close", "收盘价"),
            ("$volume", "成交量"),
            ("$open", "开盘价"),
            ("$high", "最高价"),
            ("$low", "最低价"),
            ("$vwap", "成交量加权平均价"),
            ("Mean($close, 5)", "5日均价"),
            ("Mean($close, 20)", "20日均价"),
            ("Std($close, 20)", "20日收盘价标准差"),
            ("($close / Ref($close, 1)) - 1", "日收益率"),
        ]
        
        print(f"🔍 测试基础特征性能")
        print(f"📊 股票数量: {len(instruments)}")
        print(f"📅 时间范围: {start_time} 到 {end_time}")
        print("=" * 80)
        
        results = []
        
        for i, (feature_expr, feature_name) in enumerate(basic_features):
            print(f"\n[{i+1:2d}/{len(basic_features)}] 测试: {feature_name}")
            print(f"   表达式: {feature_expr}")
            
            result = self._test_single_expression(
                expression=feature_expr,
                name=feature_name,
                instruments=instruments,
                start_time=start_time,
                end_time=end_time
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def test_alpha_expressions(self,
                             instruments: List[str],
                             start_time: str = "2020-01-01", 
                             end_time: str = "2020-03-01") -> List[Dict]:
        """测试Alpha表达式性能"""
        
        alpha_features = [
            ("(-1 * Corr(CSRank(Delta(Log($volume), 1)), CSRank((($close - $open) / $open)), 6))", "ALPHA1_相关性"),
            ("(-1 * Delta((($close - $low) - ($high - $close)) / ($high - $low), 1))", "ALPHA2_价格位置"),
            ("Mean($close, 5) / Mean($close, 20)", "ALPHA3_均线比值"),
            ("TSRank($volume, 20)", "ALPHA4_成交量排序"),
            ("CSRank($close / $open)", "ALPHA5_横截面涨幅"),
            ("Corr($close, $volume, 10)", "ALPHA6_价量相关性"),
            ("Std($close, 20) / Mean($close, 20)", "ALPHA7_波动率系数"),
            ("Sum($volume, 5) / Sum($volume, 20)", "ALPHA8_成交量比"),
            ("Max($high, 5) / Min($low, 5)", "ALPHA9_价格振幅"),
            ("EMA($close, 10) / EMA($close, 30)", "ALPHA10_EMA比值"),
        ]
        
        print(f"\n🚀 测试Alpha表达式性能")
        print(f"📊 股票数量: {len(instruments)}")
        print(f"📅 时间范围: {start_time} 到 {end_time}")
        print("=" * 80)
        
        results = []
        
        for i, (feature_expr, feature_name) in enumerate(alpha_features):
            print(f"\n[{i+1:2d}/{len(alpha_features)}] 测试: {feature_name}")
            print(f"   表达式: {feature_expr}")
            
            result = self._test_single_expression(
                expression=feature_expr,
                name=feature_name,
                instruments=instruments,
                start_time=start_time,
                end_time=end_time
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _test_single_expression(self,
                              expression: str,
                              name: str,
                              instruments: List[str],
                              start_time: str,
                              end_time: str,
                              timeout: float = 120.0) -> Dict:
        """测试单个表达式"""
        
        result = {
            'name': name,
            'expression': expression,
            'status': 'unknown',
            'error': None,
            'execution_time': None,
            'memory_before': None,
            'memory_after': None,
            'memory_usage': None,
            'data_shape': None,
            'instruments_count': len(instruments),
            'date_range': f"{start_time} to {end_time}",
            'data_quality': None,
            'warnings': []
        }
        
        try:
            # 记录初始内存
            gc.collect()
            result['memory_before'] = self.measure_memory_usage()
            
            # 捕获警告
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # 执行表达式计算
                start_exec = time.time()
                
                df = D.features(
                    instruments,
                    [expression],
                    start_time=start_time,
                    end_time=end_time
                )
                
                end_exec = time.time()
                
                # 记录警告
                result['warnings'] = [str(warning.message) for warning in w]
            
            # 记录执行时间
            result['execution_time'] = end_exec - start_exec
            
            # 记录数据信息
            result['data_shape'] = df.shape
            
            # 数据质量分析
            if not df.empty:
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    result['data_quality'] = {
                        'total_values': df.size,
                        'nan_count': df.isnull().sum().sum(),
                        'nan_ratio': df.isnull().sum().sum() / df.size,
                        'inf_count': np.isinf(numeric_df).sum().sum(),
                        'zero_count': (numeric_df == 0).sum().sum(),
                        'unique_instruments': df.index.get_level_values('instrument').nunique(),
                        'date_count': df.index.get_level_values('datetime').nunique(),
                        'value_stats': {
                            'min': float(numeric_df.min().min()),
                            'max': float(numeric_df.max().max()), 
                            'mean': float(numeric_df.mean().mean()),
                            'std': float(numeric_df.std().mean()),
                        }
                    }
            
            # 记录结束内存
            result['memory_after'] = self.measure_memory_usage()
            result['memory_usage'] = result['memory_after'] - result['memory_before']
            
            result['status'] = 'success'
            
            print(f"   ✅ 成功! 耗时: {result['execution_time']:.3f}s, 数据: {result['data_shape']}, 内存: +{result['memory_usage']:.1f}MB")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['memory_after'] = self.measure_memory_usage()
            if result['memory_before']:
                result['memory_usage'] = result['memory_after'] - result['memory_before']
            
            print(f"   ❌ 失败: {result['error']}")
            
            # 错误分类
            if "window must be an integer" in str(e):
                result['error_type'] = 'window_parameter_error'
            elif "unsupported operand type" in str(e):
                result['error_type'] = 'operator_error'
            elif "not found" in str(e).lower():
                result['error_type'] = 'data_not_found'
            else:
                result['error_type'] = 'unknown_error'
        
        return result
    
    def test_scalability(self,
                        base_instruments: List[str],
                        test_sizes: List[int] = None,
                        start_time: str = "2020-01-01",
                        end_time: str = "2020-02-01") -> List[Dict]:
        """测试可扩展性性能"""
        
        if test_sizes is None:
            test_sizes = [10, 30, 50, 100, len(base_instruments)]
        
        # 确保测试大小不超过可用股票数量
        test_sizes = [size for size in test_sizes if size <= len(base_instruments)]
        
        test_expression = "Mean($close, 5)"
        test_name = "可扩展性测试"
        
        print(f"\n📈 测试可扩展性性能")
        print(f"🧪 测试表达式: {test_expression}")
        print(f"📊 测试规模: {test_sizes}")
        print("=" * 80)
        
        results = []
        
        for size in test_sizes:
            instruments = base_instruments[:size]
            print(f"\n🔍 测试股票数量: {size}")
            
            result = self._test_single_expression(
                expression=test_expression,
                name=f"{test_name}_{size}股票",
                instruments=instruments,
                start_time=start_time,
                end_time=end_time
            )
            
            result['test_size'] = size
            results.append(result)
            self.results.append(result)
        
        return results
    
    def generate_csi300_report(self, results: List[Dict] = None) -> str:
        """生成CSI300专用性能报告"""
        
        results = results or self.results
        
        if not results:
            return "没有测试结果"
        
        # 分类结果
        basic_results = [r for r in results if '测试' not in r['name'] and 'ALPHA' not in r['name']]
        alpha_results = [r for r in results if 'ALPHA' in r['name']]
        scalability_results = [r for r in results if '测试' in r['name']]
        
        # 统计信息
        total_tests = len(results)
        successful_tests = len([r for r in results if r['status'] == 'success'])
        failed_tests = total_tests - successful_tests
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("📊 CSI300 数据性能测试报告")
        report.append("=" * 80)
        
        # 总体统计
        report.append(f"\n📈 总体统计:")
        report.append(f"   总测试数量: {total_tests}")
        report.append(f"   成功数量: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        report.append(f"   失败数量: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # 性能分析
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            execution_times = [r['execution_time'] for r in successful_results]
            memory_usage = [r['memory_usage'] for r in successful_results if r['memory_usage'] is not None]
            
            report.append(f"\n⏱️  执行时间分析:")
            report.append(f"   平均执行时间: {np.mean(execution_times):.3f}s")
            report.append(f"   最快: {np.min(execution_times):.3f}s")
            report.append(f"   最慢: {np.max(execution_times):.3f}s")
            report.append(f"   中位数: {np.median(execution_times):.3f}s")
            
            if memory_usage:
                report.append(f"\n💾 内存使用分析:")
                report.append(f"   平均内存增长: {np.mean(memory_usage):.2f} MB")
                report.append(f"   最大内存增长: {np.max(memory_usage):.2f} MB")
                report.append(f"   最小内存增长: {np.min(memory_usage):.2f} MB")
        
        # 基础特征性能
        if basic_results:
            report.append(f"\n🔍 基础特征性能:")
            report.append("-" * 60)
            for result in basic_results:
                if result['status'] == 'success':
                    report.append(f"   ✅ {result['name']:15s}: {result['execution_time']:.3f}s, {result['data_shape']}")
                else:
                    report.append(f"   ❌ {result['name']:15s}: {result['error']}")
        
        # Alpha表达式性能
        if alpha_results:
            report.append(f"\n🚀 Alpha表达式性能:")
            report.append("-" * 60)
            for result in alpha_results:
                if result['status'] == 'success':
                    report.append(f"   ✅ {result['name']:20s}: {result['execution_time']:.3f}s, {result['data_shape']}")
                else:
                    report.append(f"   ❌ {result['name']:20s}: {result['error']}")
        
        # 可扩展性分析
        if scalability_results:
            report.append(f"\n📈 可扩展性分析:")
            report.append("-" * 60)
            
            success_scalability = [r for r in scalability_results if r['status'] == 'success']
            if len(success_scalability) > 1:
                sizes = [r['test_size'] for r in success_scalability]
                times = [r['execution_time'] for r in success_scalability]
                
                # 计算时间复杂度
                if len(sizes) >= 2:
                    time_growth_ratio = times[-1] / times[0]
                    size_growth_ratio = sizes[-1] / sizes[0]
                    complexity_estimate = np.log(time_growth_ratio) / np.log(size_growth_ratio)
                    
                    report.append(f"   股票数量范围: {min(sizes)} - {max(sizes)}")
                    report.append(f"   时间增长倍数: {time_growth_ratio:.2f}x")
                    report.append(f"   估计时间复杂度: O(n^{complexity_estimate:.2f})")
            
            for result in success_scalability:
                report.append(f"   {result['test_size']:3d} 股票: {result['execution_time']:.3f}s")
        
        # 数据质量分析
        quality_results = [r for r in successful_results if r['data_quality']]
        if quality_results:
            report.append(f"\n📊 数据质量分析:")
            report.append("-" * 60)
            
            total_nan_ratio = np.mean([r['data_quality']['nan_ratio'] for r in quality_results])
            avg_instruments = np.mean([r['data_quality']['unique_instruments'] for r in quality_results])
            avg_dates = np.mean([r['data_quality']['date_count'] for r in quality_results])
            
            report.append(f"   平均缺失值比例: {total_nan_ratio:.3f}")
            report.append(f"   平均股票数量: {avg_instruments:.0f}")
            report.append(f"   平均交易日数: {avg_dates:.0f}")
        
        # 优化建议
        report.append(f"\n💡 优化建议:")
        
        if failed_tests > 0:
            report.append(f"   🔧 修复 {failed_tests} 个失败的表达式")
        
        slow_tests = [r for r in successful_results if r['execution_time'] > 2.0]
        if slow_tests:
            report.append(f"   ⚡ 优化 {len(slow_tests)} 个执行时间超过2秒的表达式")
        
        high_memory_tests = [r for r in successful_results if r.get('memory_usage', 0) > 200]
        if high_memory_tests:
            report.append(f"   💾 优化 {len(high_memory_tests)} 个内存使用超过200MB的表达式")
        
        # CSI300特定建议
        report.append(f"\n📈 CSI300特定建议:")
        report.append(f"   📊 CSI300包含300只活跃股票，建议使用批量计算优化")
        report.append(f"   🕐 对于日频数据，建议缓存中间计算结果")
        report.append(f"   💻 对于复杂Alpha表达式，考虑并行计算")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "csi300_performance_report.txt"):
        """保存结果"""
        import json
        
        # 保存详细结果
        json_filename = filename.replace('.txt', '.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存报告
        report = self.generate_csi300_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 CSI300测试结果已保存:")
        print(f"   详细结果: {json_filename}")
        print(f"   性能报告: {filename}")

def main():
    """主函数"""
    
    print("🚀 CSI300 数据性能测试")
    print("=" * 80)
    
    # 创建测试器
    tester = CSI300PerformanceTester()
    
    # 获取CSI300股票池
    print("📊 获取CSI300成分股...")
    instruments = tester.get_csi300_instruments(sample_size=50)  # 采样50只股票进行测试
    print(f"✅ 获取到 {len(instruments)} 只股票")
    
    # 测试配置
    start_time = "2020-01-01"
    end_time = "2020-03-01"  # 2个月的数据用于快速测试
    
    try:
        # 1. 测试基础特征
        print(f"\n🔍 阶段1: 基础特征性能测试")
        #basic_results = tester.test_basic_features(instruments, start_time, end_time)
        
        # 2. 测试Alpha表达式
        print(f"\n🚀 阶段2: Alpha表达式性能测试")
        alpha_results = tester.test_alpha_expressions(instruments, start_time, end_time)
        
        # 3. 测试可扩展性
        print(f"\n📈 阶段3: 可扩展性性能测试")
        scalability_results = tester.test_scalability(
            instruments, 
            test_sizes=[10, 20, 30, min(50, len(instruments))],
            start_time=start_time,
            end_time="2020-01-15"  # 缩短时间用于可扩展性测试
        )
        
        # 生成和显示报告
        print(f"\n📊 生成性能报告...")
        report = tester.generate_csi300_report()
        print("\n" + report)
        
        # 保存结果
        tester.save_results()
        
        print(f"\n🎉 CSI300性能测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 