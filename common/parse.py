def to_json(d):
    
    str_out = []
    s4 = '    '
    s8 = s4*2

    for k, v in d.items():
        kv=''
        kv += '{}"{}":\n'.format(s4,k)
        f = (s8+'[{}]\n').format( "{},\n " + (s8 + "{},\n ")*(len(v)-2) + s8 + "{}")
        kv += f.format(*v)
        str_out.append(kv)

    return "{{\n{}}}".format((s4+',\n').join(str_out))
