
#ifndef TESTING_DATA_HPP
#define TESTING_DATA_HPP

using real = double;

// An array of test cases; each case contains an array of points to check the
// orientation of, and the exact value from the orientation test
constexpr auto orient2d_cases = std::array{
    std::pair{std::array{std::array<real, 2>{2.1, 4.7},
                         std::array<real, 2>{-1.76, 7.34},
                         std::array<real, 2>{-3.04, 9.07}},
              real{-3.2986}},
    std::pair{
        std::array{
            std::array<real, 2>{
                -0.5534416740277208202059000541339628398418426513671875,
                0.252389412564555382090247803716920316219329833984375},
            std::array<real, 2>{
                -0.67704155516317587881758299772627651691436767578125,
                -0.41388853370596112579704595191287808120250701904296875},
            std::array<real, 2>{
                -0.61524161459544834951174152593011967837810516357421875,
                -0.080749560570702871853399074097978882491588592529296875}},
        real{0.0}},
    std::pair{
        std::array{std::array<real, 2>{
                       -0.136128279746752678391885638120584189891815185546875,
                       -0.0541791054594067400529411315801553428173065185546875},
                   std::array<real, 2>{
                       -0.48377064146356241192137304096831940114498138427734375,
                       -0.97690988665798794698957863147370517253875732421875},
                   std::array<real, 2>{
                       -0.48377064146356241192137304096831940114498138427734375,
                       -0.97690988665798794698957863147370517253875732421875}},
        real{0.0}},
    std::pair{
        std::array{std::array<real, 2>{
                       0.207541894308931329504730456392280757427215576171875,
                       0.1563475492240156139445161898038350045680999755859375},
                   std::array<real, 2>{
                       0.4041592280934918068879824204486794769763946533203125,
                       0.900118388435432681404790855594910681247711181640625},
                   std::array<real, 2>{
                       0.4041592280934918068879824204486794769763946533203125,
                       0.900118388435432681404790855594910681247711181640625}},
        real{0.0}},
    std::pair{
        std::array{
            std::array<real, 2>{
                -1.7171422252264145758005042807781137526035308837890625000e-01,
                -5.2510566572898098591792859224369749426841735839843750000e-01},
            std::array<real, 2>{
                -8.5437492488284516589658323937328532338142395019531250000e-01,
                -9.2080857531515714065051270154071971774101257324218750000e-01},
            std::array<real, 2>{
                -8.5437492488284516589658323937328532338142395019531250000e-01,
                -9.2080857531515714065051270154071971774101257324218750000e-01}},
        real{0.0}},
    std::pair{
        std::array{std::array<real, 2>{
                       0.3553578622027533384652997483499348163604736328125,
                       -0.480972468328285884808792616240680217742919921875},
                   std::array<real, 2>{
                       0.8114414000969174534105832208297215402126312255859375,
                       -0.62971449178105132205018890090286731719970703125},
                   std::array<real, 2>{
                       0.81144140009691756443288568334537558257579803466796875,
                       -0.62971449178105132205018890090286731719970703125}},
        real{1.65136819166595213694166726827935564544155410543480e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                -3.5464218352070331352621224141330458223819732666015625000e-01,
                -1.9172231487413637918137965243658982217311859130859375000e-01},
            std::array<real, 2>{
                -7.6892387371201198487113970259088091552257537841796875000e-01,
                -1.5316136015713999185550164838787168264389038085937500000e-01},
            std::array<real, 2>{
                -7.6892387371201187384883724007522687315940856933593750000e-01,
                -1.5316136015713996409992603275895817205309867858886718750e-01}},
        real{-1.57797527561091616335031539256741587140900588291572e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                -0.6196994772052517586047315489850006997585296630859375,
                -0.0175939567878289349067699731676839292049407958984375},
            std::array<real, 2>{
                -0.382707284932337632454846243490464985370635986328125,
                0.496344993310877224956811915035359561443328857421875},
            std::array<real, 2>{
                -0.382707284932337632454846243490464985370635986328125,
                0.496344993310877280467963146293186582624912261962890625}},
        real{1.31557094258890686727595134838684051363729519555819e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                0.2051606721897396568010663031600415706634521484375,
                -0.5053120942616100563782310928218066692352294921875},
            std::array<real, 2>{
                0.1689275307732902575708067161031067371368408203125,
                -0.4368182592273488484835297640529461205005645751953125},
            std::array<real, 2>{
                0.168927530773289980015050559813971631228923797607421875,
                -0.436818259227348126838563757701194845139980316162109375}},
        real{-7.13660593079846475416494410692059236892685229270220e-18}},
    std::pair{std::array{
                  std::array<real, 2>{
                      -0.601060587147396407914357041590847074985504150390625,
                      -0.19198343624255087558339027964393608272075653076171875},
                  std::array<real, 2>{
                      0.94947677281838327445484537747688591480255126953125,
                      0.5291726267551359885743522681877948343753814697265625},
                  std::array<real, 2>{
                      0.949476772818383718544055227539502084255218505859375,
                      0.529172626755136210618957193219102919101715087890625}},
              real{2.40308293198778675345200380518198358882962881611506e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                1.2347529353129438334235601359978318214416503906250000000e-04,
                2.0107915805777976103740911639761179685592651367187500000e-02},
            std::array<real, 2>{
                5.0070758002566106981134907982777804136276245117187500000e-02,
                -5.0371887897147826773647238951525650918483734130859375000e-01},
            std::array<real, 2>{
                5.0070758002566127797816619704462937079370021820068359375e-02,
                -5.0371887897147848978107731454656459391117095947265625000e-01}},
        real{-1.86188997256984065170061223492029039431877087458584e-19}},
    std::pair{
        std::array{
            std::array<real, 2>{
                -3.5745691950812696902062270964961498975753784179687500000e-01,
                -1.2013344828381911089110190005158074200153350830078125000e-01},
            std::array<real, 2>{
                1.1058909158000740369232062221271917223930358886718750000e-01,
                9.5179231014359721996243024477735161781311035156250000000e-01},
            std::array<real, 2>{
                1.1058909158000738981453281439826241694390773773193359375e-01,
                9.5179231014359710894012778226169757544994354248046875000e-01}},
        real{-3.70875875882145468583064100441501606009188028091508e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                -7.5633482372098181922837056845310144126415252685546875000e-01,
                2.4052759075689933432329326024046167731285095214843750000e-01},
            std::array<real, 2>{
                1.0766055134319851838142767519457265734672546386718750000e-01,
                -5.7138221302988290162261364457663148641586303710937500000e-02},
            std::array<real, 2>{
                1.0766055134319849062585205956565914675593376159667968750e-01,
                -5.7138221302988276284473556643206393346190452575683593750e-02}},
        real{3.72845852725796468083842286416507770116517793418257e-18}},
    std::pair{
        std::array{
            std::array<real, 2>{
                1.5161523499642082235538964596344158053398132324218750000e-01,
                9.0944015010502865514752102171769365668296813964843750000e-01},
            std::array<real, 2>{
                9.2985026990994645856858369370456784963607788085937500000e-02,
                4.9369177124034790971052188979228958487510681152343750000e-01},
            std::array<real, 2>{
                9.2985026990994659734646177184913540259003639221191406250e-02,
                4.9369177124034796522167312105011660605669021606445312500e-01}},
        real{2.51503744001757622354534284993911166137213735599643e-18}},
    std::pair{
        std::array{
            std::array<real, 2>{
                5.3676019520491058401034933922346681356430053710937500000e-01,
                -6.9620638482637020594268051354447379708290100097656250000e-01},
            std::array<real, 2>{
                -3.4844641244844209992947980936150997877120971679687500000e-01,
                -8.4720631310135408043038296455051749944686889648437500000e-02},
            std::array<real, 2>{
                -3.4844641244844209992947980936150997877120971679687500000e-01,
                -8.4720631310135421920826104269508505240082740783691406250e-02}},
        real{1.22847094670884932612862798310262689762616152553722e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                2.3526394694766694648535576561698690056800842285156250000e-01,
                2.7417890132415534409915380820166319608688354492187500000e-01},
            std::array<real, 2>{
                5.7393348266317767425448437279555946588516235351562500000e-01,
                4.7643906214362630180403357371687889099121093750000000000e-01},
            std::array<real, 2>{
                5.7393348266317778527678683531121350824832916259765625000e-01,
                4.7643906214362635731518480497470591217279434204101562500e-01}},
        real{-3.65545293609277275551810570130333435898144803189933e-18}},
    std::pair{
        std::array{
            std::array<real, 2>{
                3.9806574835726160621618419099831953644752502441406250000e-01,
                1.8191956024739375230581117648398503661155700683593750000e-01},
            std::array<real, 2>{
                -6.7505004460570494639881644616252742707729339599609375000e-01,
                8.3018308097202231365940860996488481760025024414062500000e-02},
            std::array<real, 2>{
                -6.7505004460570505742111890867818146944046020507812500000e-01,
                8.3018308097202231365940860996488481760025024414062500000e-02}},
        real{-1.09802447301400897194801137821125796957857348370297e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                3.7447843078873455091581945453071966767311096191406250000e-01,
                -2.0698566428523490490221092841238714754581451416015625000e-01},
            std::array<real, 2>{
                -3.8901702824466000141256927236099727451801300048828125000e-01,
                5.7779358452816964586418180260807275772094726562500000000e-01},
            std::array<real, 2>{
                -3.8901702824465989039026680984534323215484619140625000000e-01,
                5.7779358452816953484187934009241871535778045654296875000e-01}},
        real{-2.36297534650486956020509811446294746457994688602217e-18}},
    std::pair{
        std::array{
            std::array<real, 2>{
                -1.4914977747016322506823371440987102687358856201171875000e-01,
                1.6460537779426087645617826638044789433479309082031250000e-01},
            std::array<real, 2>{
                -9.1231299987927949590726939277374185621738433837890625000e-01,
                7.5822887917621817344127066462533548474311828613281250000e-01},
            std::array<real, 2>{
                -9.1231299987927971795187431780504994094371795654296875000e-01,
                7.5822887917621839548587558965664356946945190429687500000e-01}},
        real{-3.76453803745755375085176375458091955062901636918510e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                6.5880007529389628118110522336792200803756713867187500000e-01,
                2.1681646947284205495520836848299950361251831054687500000e-01},
            std::array<real, 2>{
                -6.5012535843874286189958411341649480164051055908203125000e-01,
                -1.8018973948220062819558506816974841058254241943359375000e-01},
            std::array<real, 2>{
                -6.5012535843874219576576933832257054746150970458984375000e-01,
                -1.8018973948220043390655575876735383644700050354003906250e-01}},
        real{1.01494085023176692798634635543650959459114800701980e-17}},
    std::pair{
        std::array{
            std::array<real, 2>{
                9.0231184047855150787142974877497181296348571777343750000e-01,
                1.5012582471856594779069382639136165380477905273437500000e-01},
            std::array<real, 2>{
                9.4647504072471799752008791983826085925102233886718750000e-01,
                8.7198038900141061624537996976869180798530578613281250000e-01},
            std::array<real, 2>{
                9.4647504072471810854239038235391490161418914794921875000e-01,
                8.7198038900141061624537996976869180798530578613281250000e-01}},
        real{-8.01419557697574301230110264545969560790491230005059e-17}},
};

#endif // TESTING_DATA_HPP
