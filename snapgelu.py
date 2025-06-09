#--- Our custom activation ---
def snap_gelu (x , k : float = 1.5 , b : float = 0.5) :
     return x * torch . sigmoid ( k * x + b )
class SnapGELU ( nn . Module ) :
    def __init__ ( self , k : float = 1.5 , b : float = 0.5) :
     super () . __init__ ()
     self . k = k
     self . b = b
def forward ( self , x ) :
    return snap_gelu (x , self .k , self . b )