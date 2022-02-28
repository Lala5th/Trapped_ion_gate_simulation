#!/usr/bin/python3
import json
import sys
import re

args = sys.argv[1:]

regex_patterns = [
    ("Base\(\(([^\,]*)\,[\ ]*([^\)]*)\)\)",         "\\1"),
    ("Conj\(([^\)]*)\)",                            "Conj[\\1]"),
    ("Integral\(([^\)]*)\)",                        "Brace[\\1]"),
    ("Mult\(\(([^\)]*)\)\)",                        "FMult[\\1]"),
    ("1","a"),("2","b")
]

def get_term(scalar_sum_element):
    coeff = f"%s(%i/%i)" % ("I*" if scalar_sum_element["imaginary"] == 1 else "",scalar_sum_element["fraction"][0],scalar_sum_element["fraction"][1])
    cond = scalar_sum_element['function']
    prev = ""
    while prev != cond:
        prev = cond
        for pattern in regex_patterns:
            cond = re.sub(pattern[0],pattern[1],cond)

    return coeff + "*" + cond

if __name__ == "__main__":
    with open(args[0], "r") as fp:
        data = json.load(fp)
    op = {}
    for transform in data:
        for term_sum in transform["terms"]:
            term = ""
            for scalar_sum in term_sum["scalar"]:
                term += "+1*" + get_term(scalar_sum) + "[#]"
            op.setdefault((term_sum['op']['Sy'],term_sum['op']['create'],term_sum['op']['destroy']),"")
            op[(term_sum['op']['Sy'],term_sum['op']['create'],term_sum['op']['destroy'])] += term
    for o in op:
        op[o] += "&"

    with open(args[1],'w') as file:
        file.write("DrivingConditions:=Function[{a,b}, #[Subscript[t,g]]]&/@ {\n")
        for k, v in op.items():
            file.write(v + ",\n\n")
        file.write("}")