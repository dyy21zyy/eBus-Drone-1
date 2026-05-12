from pathlib import Path

def _parse_simple_yaml(txt:str):
    out={}
    stack=[out]; ind=[0]
    for raw in txt.splitlines():
        if not raw.strip() or raw.strip().startswith('#'): continue
        i=len(raw)-len(raw.lstrip(' ')); line=raw.strip()
        while i<ind[-1]: stack.pop(); ind.pop()
        if ':' in line:
            k,v=line.split(':',1);k=k.strip();v=v.strip()
            if not v:
                d={}; stack[-1][k]=d; stack.append(d); ind.append(i+2)
            else:
                if v.startswith('[') and v.endswith(']'):
                    val=[_coerce(x.strip()) for x in v[1:-1].split(',') if x.strip()]
                else: val=_coerce(v)
                stack[-1][k]=val
    return out

def _coerce(v):
    if v in ('true','True'): return True
    if v in ('false','False'): return False
    try:
        return int(v) if '.' not in v else float(v)
    except: return v

def load_yaml(path):
    return _parse_simple_yaml(Path(path).read_text())

def load_config(root:Path,instance='medium'):
    cfg=load_yaml(root/'configs/default.yaml')
    cfg['instance_cfg']=load_yaml(root/f'configs/instances/{instance}.yaml')
    return cfg
