"?+
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1fffff??@Afffff??@a????k??i????k???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff?@9fffff?@Afffff?@Ifffff?@an	?????i42??_???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1333339?@9333339?@A333339?@I333339?@a^??+#??i	???????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333E?@933333E?@A33333E?@I33333E?@a?A?Ij??iB??(A???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(133333?@933333?@A33333?@I33333?@a?2?&8??i??ݯ?????Unknown
^HostGatherV2"GatherV2(133333K}@933333K}@A33333K}@I33333K}@a??Ǥ?z??i?q*?,???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffffg@9fffffg@Afffffg@Ifffffg@a??7EД?iv0T?????Unknown
?	HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?????9f@9?????9f@A?????9f@I?????9f@af}ݒ???ia???s???Unknown
t
Host_FusedMatMul"sequential/dense_1/BiasAdd(1?????\a@9?????\a@A?????\a@I?????\a@a???Շc??i??A?a????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?????y]@9?????y]@A?????y]@I?????y]@a?0ӧ???ik`?_?[???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?????\@9?????\@A?????\@I?????\@a(5]?Z??i@]?_????Unknown
wHost	LeakyRelu" sequential/leaky_re_lu/LeakyRelu(1?????U@9?????U@A?????U@I?????U@a1????i? E|???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1fffffFS@9fffffFS@AfffffFS@IfffffFS@a?e0?Sl??i?ܓ-S???Unknown
?HostLeakyReluGrad"<gradient_tape/sequential/leaky_re_lu/LeakyRelu/LeakyReluGrad(1?????lR@9?????lR@A?????lR@I?????lR@a?k?????iʃ?̕???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1??????Q@9??????Q@A??????Q@I??????Q@a????r??iܐ:o????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(133333?L@933333?L@A33333?L@I33333?L@aL?$++z?i??FW?	???Unknown
`HostGatherV2"
GatherV2_1(1???????@9???????@A???????@I???????@a??????l?i'a8?&???Unknown
oHostSoftmax"sequential/dense_1/Softmax(133333?:@933333?:@A33333?:@I33333?:@asƆ??"h?i????>???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333?:@933333?:@A      6@I      6@a?????c?i??l?R???Unknown
ZHostArgMax"ArgMax(133333?1@933333?1@A33333?1@I33333?1@a???_?i??J?b???Unknown
dHostDataset"Iterator::Model(1fffff?:@9fffff?:@Affffff.@Iffffff.@a??hy?z[?i?^H?Fp???Unknown
\HostArgMax"ArgMax_1(1??????-@9??????-@A??????-@I??????-@a9ǔ?Z?i???n?}???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333)@9333333)@A333333)@I333333)@a?grf?V?i???!????Unknown
iHostWriteSummary"WriteSummary(1333333(@9333333(@A333333(@I333333(@a-~,???U?i??N!?????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff'@9ffffff'@Affffff'@Iffffff'@a?)? ?&U?i??ސ?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?????L0@9?????L0@A??????$@I??????$@aÖ,8?R?iI??,?????Unknown
YHostPow"Adam/Pow(1      $@9      $@A      $@I      $@a2B?jR?ij?"9?????Unknown
gHostStridedSlice"strided_slice(1ffffff#@9ffffff#@Affffff#@Iffffff#@aĂ?@?Q?i?zu?ù???Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1333333#@9333333#@A333333#@I333333#@a??w??ZQ?i???Uq????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @aߏYbUOI?iM*+?????Unknown
? HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a?grf?F?i`??w????Unknown
?!HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a??h'UE?i?.?N?????Unknown
["HostAddV2"Adam/add(1333333@9333333@A333333@I333333@aw?;4??D?iu=Wt
????Unknown
x#HostDataset"#Iterator::Model::ParallelMapV2::Zip(1?????YI@9?????YI@Affffff@Iffffff@azl?C?pB?iP=h??????Unknown
e$Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a??w??ZA?iK[?\?????Unknown?
?%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@aX?J?h?@?i?-??<????Unknown
V&HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@aF?w??=?i?<@??????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?6???<?i? ?W?????Unknown
{(HostSum"*categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a9ǔ?:?i[?T??????Unknown
?)HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@a9ǔ?:?i"????????Unknown
[*HostPow"
Adam/Pow_1(1??????@9??????@A??????@I??????@aq??u:?i?Z?ˀ????Unknown
~+HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1??????	@9??????	@A??????	@I??????	@a+?JK?#7?i??YJe????Unknown
?,HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a?=???j6?i??,?2????Unknown
?-HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff?2@9fffff?2@A??????@I??????@aÖ,8?2?i??0L?????Unknown
a.HostIdentity"Identity(1      ??9      ??A      ??I      ??a?6????i      ???Unknown?*?*
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff?@9fffff?@Afffff?@Ifffff?@a??????i???????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1333339?@9333339?@A333339?@I333339?@a?k???f??i?;je>????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333E?@933333E?@A33333E?@I33333E?@a?	{??in??5?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(133333?@933333?@A33333?@I33333?@aj???;??i{???SB???Unknown
^HostGatherV2"GatherV2(133333K}@933333K}@A33333K}@I33333K}@a>?h???i<??? ????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffffg@9fffffg@Afffffg@Ifffffg@a??,)rH??i
(zd????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?????9f@9?????9f@A?????9f@I?????9f@a?n?12??i???????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1?????\a@9?????\a@A?????\a@I?????\a@a??/????i?iO??g???Unknown
{	HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?????y]@9?????y]@A?????y]@I?????y]@a??X???i???????Unknown
?
HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?????\@9?????\@A?????\@I?????\@a[?	???i?7??M????Unknown
wHost	LeakyRelu" sequential/leaky_re_lu/LeakyRelu(1?????U@9?????U@A?????U@I?????U@a??MƂ???io??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1fffffFS@9fffffFS@AfffffFS@IfffffFS@a?Z^?#0??im?st{???Unknown
?HostLeakyReluGrad"<gradient_tape/sequential/leaky_re_lu/LeakyRelu/LeakyReluGrad(1?????lR@9?????lR@A?????lR@I?????lR@aȒ??~??i?v?n?????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1??????Q@9??????Q@A??????Q@I??????Q@a??fZ????i?+?<????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(133333?L@933333?L@A33333?L@I33333?L@a =?(????i??? ?????Unknown
`HostGatherV2"
GatherV2_1(1???????@9???????@A???????@I???????@a?1?(C?u?i? ?????Unknown
oHostSoftmax"sequential/dense_1/Softmax(133333?:@933333?:@A33333?:@I33333?:@a3??^U#r?i?
?Qb????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333?:@933333?:@A      6@I      6@a r q??m?i>OAF????Unknown
ZHostArgMax"ArgMax(133333?1@933333?1@A33333?1@I33333?1@a??^lTh?i?i??R???Unknown
dHostDataset"Iterator::Model(1fffff?:@9fffff?:@Affffff.@Iffffff.@a?2*?ʦd?i?f`?'???Unknown
\HostArgMax"ArgMax_1(1??????-@9??????-@A??????-@I??????-@a??Jc?d?i???
<???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333)@9333333)@A333333)@I333333)@a????xa?i?ܡ?3M???Unknown
iHostWriteSummary"WriteSummary(1333333(@9333333(@A333333(@I333333(@a+?&~?p`?ig ?]???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff'@9ffffff'@Affffff'@Iffffff'@ap?l??_?i?JV??m???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?????L0@9?????L0@A??????$@I??????$@a?*b??B\?i???˪{???Unknown
YHostPow"Adam/Pow(1      $@9      $@A      $@I      $@a{P?	N,[?i\ͧ?@????Unknown
gHostStridedSlice"strided_slice(1ffffff#@9ffffff#@Affffff#@Iffffff#@a?,ԝ?[Z?ir?v?n????Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1333333#@9333333#@A333333#@I333333#@a8v?yZ?i??3?y????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a????iS?i?o}?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a????xQ?i?n???????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@aS?H8P?i$??Տ????Unknown
[ HostAddV2"Adam/add(1333333@9333333@A333333@I333333@a???HP?O?iҵ?)q????Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1?????YI@9?????YI@Affffff@Iffffff@a???Qn?K?i?T_????Unknown
e"Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a8v?yJ?i?????????Unknown?
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@a	2??I?i??CG????Unknown
V$HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@aPzȂ+HF?i ??N?????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a/?:?E?iC}??H????Unknown
{&HostSum"*categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a??Jc?D?i?OG|O????Unknown
?'HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@a??Jc?D?i?"?fV????Unknown
[(HostPow"
Adam/Pow_1(1??????@9??????@A??????@I??????@a?Xk??C?i{?f?:????Unknown
~)HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1??????	@9??????	@A??????	@I??????	@a&???dA?i?????????Unknown
?*HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a7???@?ir????????Unknown
?+HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff?2@9fffff?2@A??????@I??????@a?*b??B<?i?(?R????Unknown
a,HostIdentity"Identity(1      ??9      ??A      ??I      ??a/?:??i?????????Unknown?