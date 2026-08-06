import re

def gen():
    out = []
    
    # E is shape (3,3). d is (4,3).
    # xl is d @ E
    for i in range(4):
        for j, c in enumerate(['x', 'y', 'z']):
            out.append(f"d{i}{c} = xe[e, {i}, {j}] - xe[e, 0, {j}]")
    
    for i in range(4):
        for j, c in enumerate(['x', 'y', 'z']):
            terms = [f"d{i}{k} * E[e, {ki}, {j}]" for ki, k in enumerate(['x', 'y', 'z'])]
            out.append(f"xl{i}{c} = {' + '.join(terms)}")
            
    # cx = xl - xl.mean
    for c in ['x', 'y']:
        out.append(f"mean_{c} = 0.25 * (xl0{c} + xl1{c} + xl2{c} + xl3{c})")
        for i in range(4):
            out.append(f"c{i}{c} = xl{i}{c} - mean_{c}")
            
    # zl1 = -mean(z)
    out.append("zl1[e] = -0.25 * (xl0z + xl1z + xl2z + xl3z)")
    
    out.append("x13 = 0.5 * (c0x - c2x)")
    out.append("x24 = 0.5 * (c1x - c3x)")
    out.append("y13 = 0.5 * (c0y - c2y)")
    out.append("y24 = 0.5 * (c1y - c3y)")
    out.append("l13 = x13**2 + y13**2")
    out.append("l24 = x24**2 + y24**2")
    out.append("ll[e] = l13 if l13 > l24 else l24")
    out.append("lm1 = abs(c1x*c3y - c1y*c3x)")
    out.append("lm2 = abs(c0x*c2y - c0y*c2x)")
    out.append("lm[e] = lm1 if lm1 > lm2 else lm2")
    
    return "\n".join(out)

if __name__ == "__main__":
    print(gen())
