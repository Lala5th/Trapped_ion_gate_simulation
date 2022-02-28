DrivingConditions=Function[{a,b}, #[Subscript[t,g]]]&/@ {
+1*(-1/1)*Brace[a][#]+1*(4/1)*Brace[FMult[a, Brace[a], Brace[Conj[a]]]][#]+1*(-2/1)*Brace[FMult[Conj[a], Brace[a], Brace[a]]][#]+1*(-2/1)*Brace[FMult[b, Brace[FMult[Conj[b], Brace[a]]]]][#]+1*(2/1)*Brace[FMult[Conj[b], Brace[a], Brace[b]]][#]&,

+1*(1/1)*Brace[Conj[a]][#]+1*(2/1)*Brace[FMult[a, Brace[Conj[a]], Brace[Conj[a]]]][#]+1*(-4/1)*Brace[FMult[Conj[a], Brace[a], Brace[Conj[a]]]][#]+1*(-2/1)*Brace[FMult[b, Brace[Conj[a]], Brace[Conj[b]]]][#]+1*(2/1)*Brace[FMult[Conj[b], Brace[FMult[b, Brace[Conj[a]]]]]][#]&,

+1*(-1/2)*Brace[FMult[a, Brace[Conj[a]]]][#]+1*(1/2)*Brace[FMult[Conj[a], Brace[a]]][#]+1*(2/1)*Brace[FMult[a, Brace[a], Brace[Conj[a]], Brace[Conj[a]]]][#]+1*(-2/1)*Brace[FMult[Conj[a], Brace[a], Brace[a], Brace[Conj[a]]]][#]+1*(-1/4)*Brace[FMult[b, Brace[Conj[b]]]][#]+1*(-2/1)*Brace[FMult[b, Brace[Conj[a]], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*(1/4)*Brace[FMult[Conj[b], Brace[b]]][#]+1*(2/1)*Brace[FMult[Conj[b], Brace[a], Brace[FMult[b, Brace[Conj[a]]]]]][#]&,

+1*I*(-1/2)*Brace[b][#]+1*I*(2/1)*Brace[FMult[b, Brace[a], Brace[Conj[a]]]][#]+1*I*(2/1)*Brace[FMult[a, Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(4/1)*Brace[FMult[a, Brace[b], Brace[Conj[a]]]][#]+1*I*(-4/1)*Brace[FMult[Conj[a], Brace[a], Brace[b]]][#]+1*I*(-4/3)*Brace[FMult[b, Brace[b], Brace[Conj[b]]]][#]+1*I*(4/3)*Brace[FMult[Conj[b], Brace[b], Brace[b]]][#]&,

+1*I*(-1/2)*Brace[Conj[b]][#]+1*I*(2/1)*Brace[FMult[Conj[b], Brace[a], Brace[Conj[a]]]][#]+1*I*(-4/1)*Brace[FMult[a, Brace[Conj[a]], Brace[Conj[b]]]][#]+1*I*(4/1)*Brace[FMult[Conj[a], Brace[a], Brace[Conj[b]]]][#]+1*I*(2/1)*Brace[FMult[Conj[a], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(4/3)*Brace[FMult[b, Brace[Conj[b]], Brace[Conj[b]]]][#]+1*I*(-4/3)*Brace[FMult[Conj[b], Brace[b], Brace[Conj[b]]]][#]&,

+1*I*(-1/1)*Brace[FMult[Conj[b], Brace[a]]][#]+1*I*(2/3)*Brace[FMult[b, Brace[Conj[a]], Brace[Conj[a]], Brace[Conj[a]]]][#]+1*I*(2/1)*Brace[FMult[Conj[b], Brace[a], Brace[a], Brace[Conj[a]]]][#]+1*I*(-1/2)*Brace[FMult[a, Brace[Conj[b]]]][#]+1*I*(-4/1)*Brace[FMult[a, Brace[a], Brace[Conj[a]], Brace[Conj[b]]]][#]+1*I*(2/1)*Brace[FMult[Conj[a], Brace[a], Brace[a], Brace[Conj[b]]]][#]+1*I*(-4/1)*Brace[FMult[a, Brace[Conj[a]], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(4/1)*Brace[FMult[Conj[a], Brace[a], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(-4/1)*Brace[FMult[Conj[a], Brace[Conj[a]], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(-4/3)*Brace[FMult[Conj[b], Brace[a], Brace[b], Brace[Conj[b]]]][#]+1*I*(8/3)*Brace[FMult[b, Brace[Conj[b]], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(-4/3)*Brace[FMult[Conj[b], Brace[b], Brace[FMult[Conj[b], Brace[a]]]]][#]&,

+1*I*(-1/1)*Brace[FMult[b, Brace[Conj[a]]]][#]+1*I*(2/1)*Brace[FMult[b, Brace[a], Brace[Conj[a]], Brace[Conj[a]]]][#]+1*I*(2/3)*Brace[FMult[Conj[b], Brace[a], Brace[a], Brace[a]]][#]+1*I*(-4/1)*Brace[FMult[a, Brace[a], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(-1/2)*Brace[FMult[Conj[a], Brace[b]]][#]+1*I*(4/1)*Brace[FMult[a, Brace[Conj[a]], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(-4/1)*Brace[FMult[Conj[a], Brace[a], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(2/1)*Brace[FMult[a, Brace[b], Brace[Conj[a]], Brace[Conj[a]]]][#]+1*I*(-4/1)*Brace[FMult[Conj[a], Brace[a], Brace[b], Brace[Conj[a]]]][#]+1*I*(-4/3)*Brace[FMult[b, Brace[b], Brace[Conj[a]], Brace[Conj[b]]]][#]+1*I*(-4/3)*Brace[FMult[b, Brace[Conj[b]], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(8/3)*Brace[FMult[Conj[b], Brace[b], Brace[FMult[b, Brace[Conj[a]]]]]][#]&,

+1*I*(-2/1)*Brace[FMult[b, Brace[Conj[a]], Brace[Conj[a]]]][#]+1*I*(-2/1)*Brace[FMult[Conj[b], Brace[a], Brace[a]]][#]+1*I*(8/3)*Brace[FMult[b, Brace[a], Brace[Conj[a]], Brace[Conj[a]], Brace[Conj[a]]]][#]+1*I*(8/3)*Brace[FMult[Conj[b], Brace[a], Brace[a], Brace[a], Brace[Conj[a]]]][#]+1*I*(-2/1)*Brace[FMult[a, Brace[a], Brace[Conj[b]]]][#]+1*I*(-16/1)*Brace[FMult[a, Brace[a], Brace[Conj[a]], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(8/1)*Brace[FMult[Conj[a], Brace[a], Brace[a], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(-2/1)*Brace[FMult[Conj[a], Brace[b], Brace[Conj[a]]]][#]+1*I*(8/1)*Brace[FMult[a, Brace[Conj[a]], Brace[Conj[a]], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(-16/1)*Brace[FMult[Conj[a], Brace[a], Brace[Conj[a]], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(16/3)*Brace[FMult[b, Brace[FMult[Conj[b], Brace[a]]], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(-16/3)*Brace[FMult[Conj[b], Brace[a], Brace[b], Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(-16/3)*Brace[FMult[b, Brace[Conj[a]], Brace[Conj[b]], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(16/3)*Brace[FMult[Conj[b], Brace[FMult[b, Brace[Conj[a]]]], Brace[FMult[b, Brace[Conj[a]]]]]][#]&,

+1*(1/2)*Brace[a][#]&,

+1*(-1/2)*Brace[Conj[a]][#]&,

+1*(1/2)*Brace[FMult[a, Brace[a]]][#]&,

+1*(-1/2)*Brace[FMult[Conj[a], Brace[Conj[a]]]][#]&,

+1*(1/1)*Brace[FMult[a, Brace[Conj[a]]]][#]+1*(-1/1)*Brace[FMult[Conj[a], Brace[a]]][#]+1*(-1/2)*Brace[FMult[b, Brace[Conj[b]]]][#]+1*(1/2)*Brace[FMult[Conj[b], Brace[b]]][#]&,

+1*I*(1/6)*Brace[b][#]&,

+1*I*(1/6)*Brace[Conj[b]][#]&,

+1*I*(1/6)*Brace[FMult[b, Brace[a]]][#]+1*I*(1/2)*Brace[FMult[a, Brace[b]]][#]&,

+1*I*(1/2)*Brace[FMult[Conj[b], Brace[a]]][#]+1*I*(-1/1)*Brace[FMult[a, Brace[Conj[b]]]][#]&,

+1*I*(1/2)*Brace[FMult[b, Brace[Conj[a]]]][#]+1*I*(-1/1)*Brace[FMult[Conj[a], Brace[b]]][#]&,

+1*I*(1/6)*Brace[FMult[Conj[b], Brace[Conj[a]]]][#]+1*I*(1/2)*Brace[FMult[Conj[a], Brace[Conj[b]]]][#]&,

+1*I*(2/1)*Brace[FMult[b, Brace[Conj[a]], Brace[Conj[a]]]][#]+1*I*(2/1)*Brace[FMult[Conj[b], Brace[a], Brace[a]]][#]+1*I*(-4/1)*Brace[FMult[a, Brace[a], Brace[Conj[b]]]][#]+1*I*(-4/1)*Brace[FMult[a, Brace[FMult[Conj[b], Brace[a]]]]][#]+1*I*(-4/1)*Brace[FMult[Conj[a], Brace[FMult[b, Brace[Conj[a]]]]]][#]+1*I*(-4/1)*Brace[FMult[Conj[a], Brace[b], Brace[Conj[a]]]][#]&,

}