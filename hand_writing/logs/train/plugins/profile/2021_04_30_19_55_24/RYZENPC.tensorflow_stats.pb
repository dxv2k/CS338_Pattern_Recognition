"?.
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1?????AA?????Aa4??S??i4??S???Unknown
mHostDeleteIterator"DeleteIterator(
1    ?/?@9fffff?~@A    ?/?@Ifffff?~@a????I??i?r?]???Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(133333?@933333?@A33333?@I33333?@al*\?	???i??(2iw???Unknown?
sHostDestroyResourceOp"DestroyResourceOp(133333?@9%?S?<?U@A33333?@I%?S?<?U@a??4??ȃ?iV?=?????Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1fffff?w@9fffff?w@Afffff?w@Ifffff?w@aN?Z?_?i?E?=]????Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1     ?l@9     ?l@A     ?l@I     ?l@a{ўg?S?iLX?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      a@9      a@A      a@I      a@a???\??F?ivT/??????Unknown
	HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1fffffS@9fffffS@AfffffS@IfffffS@a?S:?9?i ?68?????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      N@9      N@A      N@I      N@a:?/??3?i'?|? ????Unknown
^HostGatherV2"GatherV2(1?????YM@9?????YM@A?????YM@I?????YM@a??A??W3?id?l??????Unknown
vHost_FusedMatMul"sequential_1/dense_3/BiasAdd(1?????G@9?????G@A?????G@I?????G@a???r?r.?i?"?s????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(133333sF@933333sF@A33333sF@I33333sF@a6{p?-?i??ňL????Unknown
{Host	LeakyRelu"$sequential_1/leaky_re_lu_1/LeakyRelu(1?????A@9?????A@A?????A@I?????A@a߼??Hy&?i?4O?????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1??????:@9??????:@A??????:@I??????:@a[????!?igﯛ?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????9@9??????9@A??????9@I??????9@a??G=.? ?i?Ó??????Unknown
dHostDataset"Iterator::Model(1ffffffA@9ffffffA@Affffff8@Iffffff8@a#?h? ?im???????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?;@9     ?;@A     ?7@I     ?7@a???????i}??????Unknown
?HostLeakyReluGrad"@gradient_tape/sequential_1/leaky_re_lu_1/LeakyRelu/LeakyReluGrad(1??????6@9??????6@A??????6@I??????6@aS(]??i??^?????Unknown
iHostWriteSummary"WriteSummary(133333?5@933333?5@A33333?5@I33333?5@a??c\??i?B??????Unknown?
qHostSoftmax"sequential_1/dense_3/Softmax(1      0@9      0@A      0@I      0@a?י???i?{?Q????Unknown
`HostGatherV2"
GatherV2_1(1      *@9      *@A      *@I      *@a?<??"?i?-h??????Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1??????)@9??????)@A??????)@I??????)@a^o? ?i??̻b????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      &@9      &@A      &@I      &@aY?Sy???i??*??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??????$@9??????$@A??????$@I??????$@ac˔#?j?iGo?dD????Unknown
ZHostArgMax"ArgMax(1??????"@9??????"@A??????"@I??????"@ar?????iM??????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????*@9??????*@A??????!@I??????!@a{?B??2?iXh?O????Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1333333@9333333@A333333@I333333@a???Z ??iӆ?V????Unknown
gHostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a?Y??	?i+v???????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?I@9     ?I@A      @I      @a:?/???il5???????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a?????i/I?A????Unknown?
[ HostAddV2"Adam/add(1ffffff@9ffffff@Affffff@Iffffff@aV?=됆?>i?=?|????Unknown
?!HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a]Ii?x?>i~.y??????Unknown
l"HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@an??[?N?>iV?R?????Unknown
\#HostArgMax"ArgMax_1(1??????@9??????@A??????@I??????@ar?????>iY???????Unknown
?$HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@atQx?@?>i??@cJ????Unknown
?%HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ax-ٹ?>i????y????Unknown
{&HostSum"*categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a{?B??2?>ih߱<?????Unknown
V'HostSum"Sum_2(1??????@9??????@A??????@I??????@a?Un??$?>iE@???????Unknown
?(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?י???>iy?~??????Unknown
?)HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@a??v??>i/Ǚ?%????Unknown
Y*HostPow"Adam/Pow(1??????@9??????@A??????@I??????@a?????>i???K????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?2?'f?>iu/{n????Unknown
?,HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?0@9     ?0@A??????@I??????@a??]?4X?>i0?z+?????Unknown
[-HostPow"
Adam/Pow_1(1333333@9333333@A333333@I333333@aPEσ??>iB????????Unknown
~.HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @ahM???\?>iն?????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?י???>i???3?????Unknown
a0HostIdentity"Identity(1????????9????????A????????I????????a?????>i???.?????Unknown?
?1HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aJ???v??>i      ???Unknown*?-
mHostDeleteIterator"DeleteIterator(
1    ?/?@9fffff?~@A    ?/?@Ifffff?~@a?+?e/??i?+?e/???Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(133333?@933333?@A33333?@I33333?@a??h?????iHll????Unknown?
sHostDestroyResourceOp"DestroyResourceOp(133333?@9%?S?<?U@A33333?@I%?S?<?U@a??ê???i??׶???Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1fffff?w@9fffff?w@Afffff?w@Ifffff?w@a?? ???i8?	'?????Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1     ?l@9     ?l@A     ?l@I     ?l@aΪ?l???i?#pJ?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      a@9      a@A      a@I      a@a6?Wew???i??(U???Unknown
HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1fffffS@9fffffS@AfffffS@IfffffS@aٌv???}?i?n.?2B???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      N@9      N@A      N@I      N@a[?􎓙w?i?XL?eq???Unknown
^	HostGatherV2"GatherV2(1?????YM@9?????YM@A?????YM@I?????YM@aB????w?i??1 ?????Unknown
v
Host_FusedMatMul"sequential_1/dense_3/BiasAdd(1?????G@9?????G@A?????G@I?????G@a??<?,r?i?0F'?????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(133333sF@933333sF@A33333sF@I33333sF@a?:???q?i"a=????Unknown
{Host	LeakyRelu"$sequential_1/leaky_re_lu_1/LeakyRelu(1?????A@9?????A@A?????A@I?????A@a?O?-B?j?im?O????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1??????:@9??????:@A??????:@I??????:@ax?I??d?i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????9@9??????9@A??????9@I??????9@a?Ѐz#d?iv_? +???Unknown
dHostDataset"Iterator::Model(1ffffffA@9ffffffA@Affffff8@Iffffff8@a?$???1c?i?&??R>???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?;@9     ?;@A     ?7@I     ?7@ae??t?|b?igfQ]?P???Unknown
?HostLeakyReluGrad"@gradient_tape/sequential_1/leaky_re_lu_1/LeakyRelu/LeakyReluGrad(1??????6@9??????6@A??????6@I??????6@a????a?i} l??b???Unknown
iHostWriteSummary"WriteSummary(133333?5@933333?5@A33333?5@I33333?5@a?1?a?i?QC	?s???Unknown?
qHostSoftmax"sequential_1/dense_3/Softmax(1      0@9      0@A      0@I      0@a,!Y,Y?i/??5g????Unknown
`HostGatherV2"
GatherV2_1(1      *@9      *@A      *@I      *@a?3?jtT?iI>	:?????Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1??????)@9??????)@A??????)@I??????)@a???u?KT?i?'?ǔ???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      &@9      &@A      &@I      &@aT??F}NQ?iY?gYn????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??????$@9??????$@A??????$@I??????$@aé??\P?i;?+Ü????Unknown
ZHostArgMax"ArgMax(1??????"@9??????"@A??????"@I??????"@a??2??M?i?"'?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????*@9??????*@A??????!@I??????!@a JqȰK?i?jC??????Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1333333@9333333@A333333@I333333@a???L=?H?io???????Unknown
gHostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a?x!?G?iv??S????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?I@9     ?I@A      @I      @a[?􎓙G?i??ظ?????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a????F?ip?L??????Unknown?
[HostAddV2"Adam/add(1ffffff@9ffffff@Affffff@Iffffff@aҶ0?A?i%v????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a?J?\??@?i8:??B????Unknown
l HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a"h9?75>?iea??	????Unknown
\!HostArgMax"ArgMax_1(1??????@9??????@A??????@I??????@a??2??=?i?f?????Unknown
?"HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a Y, ?<?iL-i|Z????Unknown
?#HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??%E?Q<?i????????Unknown
{$HostSum"*categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a JqȰ;?i????Z????Unknown
V%HostSum"Sum_2(1??????@9??????@A??????@I??????@a;ɐn:?i6??????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a,!Y,9?i?8=/?????Unknown
?'HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@a????I7?i??O?????Unknown
Y(HostPow"Adam/Pow(1??????@9??????@A??????@I??????@a????6?iq?+M?????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?w?T??4?i`???$????Unknown
?*HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?0@9     ?0@A??????@I??????@a?hʬ^?3?i?(?+?????Unknown
[+HostPow"
Adam/Pow_1(1333333@9333333@A333333@I333333@a?Y?'@2?iX??0?????Unknown
~,HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a$wFiow/?i?T???????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a,!Y,)?ie5mg????Unknown
a.HostIdentity"Identity(1????????9????????A????????I????????a????&?i?s???????Unknown?
?/HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a???B?"?i     ???Unknown