"?,
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1?????L?@A?????L?@a???m?o??i???m?o???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?????ɓ@9?????ɓ@A?????ɓ@I?????ɓ@a??¡???i?? ?????Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1??????z@9??????z@A??????z@I??????z@aq
Q?l??i6??;????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1     ?w@9     ?w@A     ?w@I     ?w@a????ֳ?i??	?7???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1fffff?w@9fffff?w@Afffff?w@Ifffff?w@av@?}γ?i???c۰???Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1?????,q@9?????,q@A?????,q@I?????,q@a??Fv???i?y9?R{???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333?b@933333?b@A33333?b@I33333?b@a???-|M??i?ئ??u???Unknown
	HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1??????Q@9??????Q@A??????Q@I??????Q@a)2ZZ?Ǎ?i?A??????Unknown
^
HostGatherV2"GatherV2(1?????,P@9?????,P@A?????,P@I?????,P@a???b\???ii(? ?X???Unknown
vHost_FusedMatMul"sequential_1/dense_3/BiasAdd(133333sE@933333sE@A33333sE@I33333sE@a+P?????irh ?a????Unknown
{Host	LeakyRelu"$sequential_1/leaky_re_lu_1/LeakyRelu(1fffff?D@9fffff?D@Afffff?D@Ifffff?D@a&@H??9??is?I????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1333333=@9333333=@A333333=@I333333=@a?)5?[x?iƿoe ???Unknown
?HostLeakyReluGrad"@gradient_tape/sequential_1/leaky_re_lu_1/LeakyRelu/LeakyReluGrad(1??????;@9??????;@A??????;@I??????;@a????w?i?9cD???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1?????:@9?????:@A?????:@I?????:@a??y??u?iE?-??o???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333?8@933333?8@A?????L3@I?????L3@a?(???p?i?Gy?ʏ???Unknown
qHostSoftmax"sequential_1/dense_3/Softmax(1?????L0@9?????L0@A?????L0@I?????L0@aT"<??1k?i????????Unknown
iHostWriteSummary"WriteSummary(1ffffff+@9ffffff+@Affffff+@Iffffff+@a2?	I?f?im?3??????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1fffff?0@9fffff?0@A??????(@I??????(@a?~???d?i?}އ????Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1??????(@9??????(@A??????(@I??????(@a?~???d?ikn??7????Unknown
`HostGatherV2"
GatherV2_1(1ffffff'@9ffffff'@Affffff'@Iffffff'@a??R?c?iVQ8?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      '@9      '@A      '@I      '@a
߾?/c?i`0??????Unknown
ZHostArgMax"ArgMax(1??????%@9??????%@A??????%@I??????%@ag?g/b?i??$???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??????%@9??????%@A??????%@I??????%@a?v?8?b?i>?2? 6???Unknown
gHostStridedSlice"strided_slice(1ffffff%@9ffffff%@Affffff%@Iffffff%@a??n??a?iĤ???G???Unknown
dHostDataset"Iterator::Model(1?????L5@9?????L5@A      %@I      %@a??ڏ?a?iip|NY???Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      !@9      !@A      !@I      !@aݵI%?\\?iD???g???Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@aˎ.?Z?i???t???Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a?
??X?i<??	????Unknown
YHostPow"Adam/Pow(1333333@9333333@A333333@I333333@a??Q??V?i??(?a????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1?????YI@9?????YI@A??????@I??????@a???'[V?i?Ax?????Unknown
? HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a???D?ZT?i3?)ɼ????Unknown
?!HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@a?v?8?R?i? ?!?????Unknown
?"HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@a??ͤE?Q?i???Ė????Unknown
e#Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a???|nQ?i????????Unknown?
{$HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?????P?i?KK}p????Unknown
?%HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a??lY?]O?iĦa?G????Unknown
\&HostArgMax"ArgMax_1(1      @9      @A      @I      @a?]	?N?i????????Unknown
?'HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?]	?N?iRUf?K????Unknown
V(HostSum"Sum_2(1333333@9333333@A333333@I333333@aˎ.?J?i??,x?????Unknown
?)HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@aˎ.?J?i???)O????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1??????	@9??????	@A??????	@I??????	@a?@? ?ZE?i?*3??????Unknown
[+HostPow"
Adam/Pow_1(1333333@9333333@A333333@I333333@a????_ZC?i?b{|????Unknown
~,HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1333333@9333333@A333333@I333333@a}0??+@?iu??}????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1333333@9333333@A333333@I333333@a}0??+@?i???~????Unknown
?.HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?????L3@9?????L3@A333333@I333333@a}0??+@?i????????Unknown
a/HostIdentity"Identity(1????????9????????A????????I????????a?H?A(?i     ???Unknown?*?+
uHostFlushSummaryWriter"FlushSummaryWriter(1?????ɓ@9?????ɓ@A?????ɓ@I?????ɓ@a?@U%????i?@U%?????Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1??????z@9??????z@A??????z@I??????z@aY;Sf9???i??~?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1     ?w@9     ?w@A     ?w@I     ?w@a|???}}??i?+{)b???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1fffff?w@9fffff?w@Afffff?w@Ifffff?w@a?I??q??i??t/i????Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1?????,q@9?????,q@A?????,q@I?????,q@a??Nj???i?Tyւ???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333?b@933333?b@A33333?b@I33333?b@a??wIz??i9,?{????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1??????Q@9??????Q@A??????Q@I??????Q@a?.?|nb??i?????????Unknown
^HostGatherV2"GatherV2(1?????,P@9?????,P@A?????,P@I?????,P@aW+?s?`??i?SP?0???Unknown
v	Host_FusedMatMul"sequential_1/dense_3/BiasAdd(133333sE@933333sE@A33333sE@I33333sE@a<Oń???iC?gc^????Unknown
{
Host	LeakyRelu"$sequential_1/leaky_re_lu_1/LeakyRelu(1fffff?D@9fffff?D@Afffff?D@Ifffff?D@au+|*???i??WS????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1333333=@9333333=@A333333=@I333333=@a????}??i%?'?I@???Unknown
?HostLeakyReluGrad"@gradient_tape/sequential_1/leaky_re_lu_1/LeakyRelu/LeakyReluGrad(1??????;@9??????;@A??????;@I??????;@a!?kU???i?.?!k????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1?????:@9?????:@A?????:@I?????:@a.?B??D?i(??d?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333?8@933333?8@A?????L3@I?????L3@a?? "w?i&j0?2????Unknown
qHostSoftmax"sequential_1/dense_3/Softmax(1?????L0@9?????L0@A?????L0@I?????L0@a??%??s?i??N?@???Unknown
iHostWriteSummary"WriteSummary(1ffffff+@9ffffff+@Affffff+@Iffffff+@a??!?ip?i???7???Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1fffff?0@9fffff?0@A??????(@I??????(@aU????m?i?:i?T???Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1??????(@9??????(@A??????(@I??????(@aU????m?iK???r???Unknown
`HostGatherV2"
GatherV2_1(1ffffff'@9ffffff'@Affffff'@Iffffff'@aoV?֐l?i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      '@9      '@A      '@I      '@a??m???k?if<?n????Unknown
ZHostArgMax"ArgMax(1??????%@9??????%@A??????%@I??????%@a??D?j?iu.K4????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??????%@9??????%@A??????%@I??????%@a?Ų???i?i;???????Unknown
gHostStridedSlice"strided_slice(1ffffff%@9ffffff%@Affffff%@Iffffff%@a?|s /?i?i?T???????Unknown
dHostDataset"Iterator::Model(1?????L5@9?????L5@A      %@I      %@a????(i?i?I݁????Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      !@9      !@A      !@I      !@a;8??]d?i?Ll@>%???Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a?9H?`?b?i????7???Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a?KF?a?i+?A??I???Unknown
YHostPow"Adam/Pow(1333333@9333333@A333333@I333333@a/`???J`?i????Y???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1?????YI@9?????YI@A??????@I??????@a>?F?`?i??aNj???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a'?(?A;]?iT:??x???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@a?Ų???Y?ig?2?????Unknown
? HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@a44??eY?i??ZC????Unknown
e!Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@aG7E~pX?i	c}]{????Unknown?
{"HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @ad~? ??W?iH?Fv????Unknown
?#HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a??<?ɅV?i?]?*?????Unknown
\$HostArgMax"ArgMax_1(1      @9      @A      @I      @a???jo?U?i~}?b?????Unknown
?%HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???jo?U?iP?Q?I????Unknown
V&HostSum"Sum_2(1333333@9333333@A333333@I333333@a?9H?`?R?imA?ʡ????Unknown
?'HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a?9H?`?R?i?????????Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1??????	@9??????	@A??????	@I??????	@a?x?I?N?i??(ͤ????Unknown
[)HostPow"
Adam/Pow_1(1333333@9333333@A333333@I333333@a~?D:?K?i?????????Unknown
~*HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1333333@9333333@A333333@I333333@a?Z??v G?i??o?W????Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1333333@9333333@A333333@I333333@a?Z??v G?i??%?????Unknown
?,HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?????L3@9?????L3@A333333@I333333@a?Z??v G?ip????????Unknown
a-HostIdentity"Identity(1????????9????????A????????I????????a???!Y@1?i      ???Unknown?