from enum import Enum

from utils.wm.gs_provider import GsProvider
from utils.wm.tr_provider import TrProvider
from utils.wm.prc_provider import PRCProvider
from utils.wm.tag_provider import TagProvider
from utils.wm.ringid_provider import RingIDProvider
from utils.wm.hstr_provider import HSTRProvider
from utils.wm.hsqr_provider import HSQRProvider

class WmProviders(Enum):
    GS = GsProvider
    TR = TrProvider
    PRC = PRCProvider
    TAG = TagProvider
    RID = RingIDProvider
    HSTR = HSTRProvider
    HSQR = HSQRProvider

