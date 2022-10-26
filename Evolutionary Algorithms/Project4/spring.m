function [Fspring,moveL0] = spring(l,v,k,L0)
%l is orginal length of cube, v is current cube point position, k is spring constant
% to increase speed, loop is not used
%% debug
% clc
% clear
% l = 1;
% ll= 0.9;
% k=ones(1,28);
% v=[ 0 0 0;
% 0 ll 0;
% ll ll 0;
% ll 0 0;
% 0 0 ll;
% 0 ll ll;
% ll ll ll;
% ll 0 ll;];
% L0=ones(1,28)*0.1;
%% get sine motion and force for spring and seprate into xyz direction
 l1=l;
 l2=l*(2^0.5);
 l3=l*(3^0.5);
% getdirection of spring x y and z
 lx1=(sum((v(1,:)-v(4,:)).^2))^0.5;
 x14=(v(1,1)-v(4,1));
 y14=(v(1,2)-v(4,2));
 z14=(v(1,3)-v(4,3));
 lx2=(sum((v(2,:)-v(3,:)).^2))^0.5;
 x23=(v(2,1)-v(3,1));
 y23=(v(2,2)-v(3,2));
 z23=(v(2,3)-v(3,3));
 lx3=(sum((v(5,:)-v(8,:)).^2))^0.5;
 x58=(v(5,1)-v(8,1));
 y58=(v(5,2)-v(8,2));
 z58=(v(5,3)-v(8,3));
 lx4=(sum((v(6,:)-v(7,:)).^2))^0.5;
 x67=(v(6,1)-v(7,1));
 y67=(v(6,2)-v(7,2));
 z67=(v(6,3)-v(7,3));
 ly1=(sum((v(1,:)-v(2,:)).^2))^0.5;
 x12=(v(1,1)-v(2,1));
 y12=(v(1,2)-v(2,2));
 z12=(v(1,3)-v(2,3));
 ly2=(sum((v(3,:)-v(4,:)).^2))^0.5;
 x34=(v(3,1)-v(4,1));
 y34=(v(3,2)-v(4,2));
 z34=(v(3,3)-v(4,3));
 ly3=(sum((v(5,:)-v(6,:)).^2))^0.5;
 x56=(v(5,1)-v(6,1));
 y56=(v(5,2)-v(6,2));
 z56=(v(5,3)-v(6,3));
 ly4=(sum((v(7,:)-v(8,:)).^2))^0.5;
 x78=(v(7,1)-v(8,1));
 y78=(v(7,2)-v(8,2));
 z78=(v(7,3)-v(8,3));
 lz1=(sum((v(1,:)-v(5,:)).^2))^0.5;
 x15=(v(1,1)-v(5,1));
 y15=(v(1,2)-v(5,2));
 z15=(v(1,3)-v(5,3));
 lz2=(sum((v(2,:)-v(6,:)).^2))^0.5;
 x26=(v(2,1)-v(6,1));
 y26=(v(2,2)-v(6,2));
 z26=(v(2,3)-v(6,3));
 lz3=(sum((v(3,:)-v(7,:)).^2))^0.5;
 x37=(v(3,1)-v(7,1));
 y37=(v(3,2)-v(7,2));
 z37=(v(3,3)-v(7,3));
 lz4=(sum((v(4,:)-v(8,:)).^2))^0.5;
 x48=(v(4,1)-v(8,1));
 y48=(v(4,2)-v(8,2));
 z48=(v(4,3)-v(8,3));
 % get force and sine motion of spring
 fx14=(l1-lx1)*k(1);
 fx14x=fx14/lx1*x14;
 fx14y=fx14/lx1*y14;
 fx14z=fx14/lx1*z14;
 mx14x=L0(1)/lx1*x14;
 mx14y=L0(1)/lx1*y14;
 mx14z=L0(1)/lx1*z14;

 fx23=(l1-lx2)*k(2);
 fx23x=fx23/lx2*x23;
 fx23y=fx23/lx2*y23;
 fx23z=fx23/lx2*z23;
 mx23x=L0(2)/lx2*x23;
 mx23y=L0(2)/lx2*y23;
 mx23z=L0(2)/lx2*z23;

 fx58=(l1-lx3)*k(3);
 fx58x=fx58/lx3*x58;
 fx58y=fx58/lx3*y58;
 fx58z=fx58/lx3*z58;
 mx58x=L0(3)/lx3*x58;
 mx58y=L0(3)/lx3*y58;
 mx58z=L0(3)/lx3*z58;


 fx67=(l1-lx4)*k(4);
 fx67x=fx67/lx4*x67;
 fx67y=fx67/lx4*y67;
 fx67z=fx67/lx4*z67;
 mx67x=L0(4)/lx4*x67;
 mx67y=L0(4)/lx4*y67;
 mx67z=L0(4)/lx4*z67;

 fy12=(l1-ly1)*k(5);
 fy12x=fy12/ly1*x12;
 fy12y=fy12/ly1*y12;
 fy12z=fy12/ly1*z12;
 my12x=L0(5)/ly1*x12;
 my12y=L0(5)/ly1*y12;
 my12z=L0(5)/ly1*z12;

 fy34=(l1-ly2)*k(6);
 fy34x=fy34/ly2*x34;
 fy34y=fy34/ly2*y34;
 fy34z=fy34/ly2*z34;
 my34x=L0(6)/ly2*x34;
 my34y=L0(6)/ly2*y34;
 my34z=L0(6)/ly2*z34;

 fy56=(l1-ly3)*k(7);
 fy56x=fy56/ly3*x56;
 fy56y=fy56/ly3*y56;
 fy56z=fy56/ly3*z56;
 my56x=L0(7)/ly3*x56;
 my56y=L0(7)/ly3*y56;
 my56z=L0(7)/ly3*z56;

 fy78=(l1-ly4)*k(8);
 fy78x=fy78/ly4*x78;
 fy78y=fy78/ly4*y78;
 fy78z=fy78/ly4*z78;
 my78x=L0(8)/ly4*x78;
 my78y=L0(8)/ly4*y78;
 my78z=L0(8)/ly4*z78;

 fz15=(l1-lz1)*k(9);
 fz15x=fz15/lz1*x15;
 fz15y=fz15/lz1*y15;
 fz15z=fz15/lz1*z15;
 mz15x=L0(9)/lz1*x15;
 mz15y=L0(9)/lz1*y15;
 mz15z=L0(9)/lz1*z15;

 fz26=(l1-lz2)*k(10);
 fz26x=fz26/lz2*x26;
 fz26y=fz26/lz2*y26;
 fz26z=fz26/lz2*z26;
 mz26x=L0(10)/lz2*x26;
 mz26y=L0(10)/lz2*y26;
 mz26z=L0(10)/lz2*z26;

 fz37=(l1-lz3)*k(11);
 fz37x=fz37/lz3*x37;
 fz37y=fz37/lz3*y37;
 fz37z=fz37/lz3*z37;
 mz37x=L0(11)/lz3*x37;
 mz37y=L0(11)/lz3*y37;
 mz37z=L0(11)/lz3*z37;

 fz48=(l1-lz4)*k(12);
 fz48x=fz48/lz4*x48;
 fz48y=fz48/lz4*y48;
 fz48z=fz48/lz4*z48;
 mz48x=L0(12)/lz4*x48;
 mz48y=L0(12)/lz4*y48;
 mz48z=L0(12)/lz4*z48;
 % getdirection of spring xy yz and xz

 lxy1=(sum((v(1,:)-v(3,:)).^2))^0.5;
 x13=(v(1,1)-v(3,1));
 y13=(v(1,2)-v(3,2));
 z13=(v(1,3)-v(3,3));
 lxy2=(sum((v(2,:)-v(4,:)).^2))^0.5;
 x24=(v(2,1)-v(4,1));
 y24=(v(2,2)-v(4,2));
 z24=(v(2,3)-v(4,3));
 lxy3=(sum((v(5,:)-v(7,:)).^2))^0.5;
 x57=(v(5,1)-v(7,1));
 y57=(v(5,2)-v(7,2));
 z57=(v(5,3)-v(7,3));
 lxy4=(sum((v(6,:)-v(8,:)).^2))^0.5;
 x68=(v(6,1)-v(8,1));
 y68=(v(6,2)-v(8,2));
 z68=(v(6,3)-v(8,3));
 lxz1=(sum((v(1,:)-v(8,:)).^2))^0.5;
 x18=(v(1,1)-v(8,1));
 y18=(v(1,2)-v(8,2));
 z18=(v(1,3)-v(8,3));

 lxz2=(sum((v(4,:)-v(5,:)).^2))^0.5;
 x45=(v(4,1)-v(5,1));
 y45=(v(4,2)-v(5,2));
 z45=(v(4,3)-v(5,3));

 lxz3=(sum((v(2,:)-v(7,:)).^2))^0.5;
 x27=(v(2,1)-v(7,1));
 y27=(v(2,2)-v(7,2));
 z27=(v(2,3)-v(7,3));

 lxz4=(sum((v(3,:)-v(6,:)).^2))^0.5;
 x36=(v(3,1)-v(6,1));
 y36=(v(3,2)-v(6,2));
 z36=(v(3,3)-v(6,3));

 lyz1=(sum((v(1,:)-v(6,:)).^2))^0.5;
 x16=(v(1,1)-v(6,1));
 y16=(v(1,2)-v(6,2));
 z16=(v(1,3)-v(6,3));

 lyz2=(sum((v(2,:)-v(5,:)).^2))^0.5;
 x25=(v(2,1)-v(5,1));
 y25=(v(2,2)-v(5,2));
 z25=(v(2,3)-v(5,3));

 lyz3=(sum((v(3,:)-v(8,:)).^2))^0.5;
 x38=(v(3,1)-v(8,1));
 y38=(v(3,2)-v(8,2));
 z38=(v(3,3)-v(8,3));

 lyz4=(sum((v(4,:)-v(7,:)).^2))^0.5;
 x47=(v(4,1)-v(7,1));
 y47=(v(4,2)-v(7,2));
 z47=(v(4,3)-v(7,3));
% get force and sine motion of spring
 fxy13=(l2-lxy1)*k(13);
 fxy13x=fxy13/lxy1*x13;
 fxy13y=fxy13/lxy1*y13;
 fxy13z=fxy13/lxy1*z13;
 mxy13x=L0(13)/lxy1*x13;
 mxy13y=L0(13)/lxy1*y13;
 mxy13z=L0(13)/lxy1*z13;

 fxy24=(l2-lxy2)*k(14);
 fxy24x=fxy24/lxy2*x24;
 fxy24y=fxy24/lxy2*y24;
 fxy24z=fxy24/lxy2*z24;
 mxy24x=L0(14)/lxy2*x24;
 mxy24y=L0(14)/lxy2*y24;
 mxy24z=L0(14)/lxy2*z24;

 fxy57=(l2-lxy3)*k(15);
 fxy57x=fxy57/lxy3*x57;
 fxy57y=fxy57/lxy3*y57;
 fxy57z=fxy57/lxy3*z57;
 mxy57x=L0(15)/lxy3*x57;
 mxy57y=L0(15)/lxy3*y57;
 mxy57z=L0(15)/lxy3*z57;

 fxy68=(l2-lxy4)*k(16);
 fxy68x=fxy68/lxy4*x68;
 fxy68y=fxy68/lxy4*y68;
 fxy68z=fxy68/lxy4*z68;
 mxy68x=L0(16)/lxy4*x68;
 mxy68y=L0(16)/lxy4*y68;
 mxy68z=L0(16)/lxy4*z68;

 fxz18=(l2-lxz1)*k(17);
 fxz18x=fxz18/lxz1*x18;
 fxz18y=fxz18/lxz1*y18;
 fxz18z=fxz18/lxz1*z18;
 mxz18x=L0(17)/lxz1*x18;
 mxz18y=L0(17)/lxz1*y18;
 mxz18z=L0(17)/lxz1*z18;

 fxz45=(l2-lxz2)*k(18);
 fxz45x=fxz45/lxz2*x45;
 fxz45y=fxz45/lxz2*y45;
 fxz45z=fxz45/lxz2*z45;
 mxz45x=L0(18)/lxz2*x45;
 mxz45y=L0(18)/lxz2*y45;
 mxz45z=L0(18)/lxz2*z45;

 fxz27=(l2-lxz3)*k(19);
 fxz27x=fxz27/lxz3*x27;
 fxz27y=fxz27/lxz3*y27;
 fxz27z=fxz27/lxz3*z27;
 mxz27x=L0(19)/lxz3*x27;
 mxz27y=L0(19)/lxz3*y27;
 mxz27z=L0(19)/lxz3*z27;

 fxz36=(l2-lxz4)*k(20);
 fxz36x=fxz36/lxz4*x36;
 fxz36y=fxz36/lxz4*y36;
 fxz36z=fxz36/lxz4*z36;
 mxz36x=L0(20)/lxz4*x36;
 mxz36y=L0(20)/lxz4*y36;
 mxz36z=L0(20)/lxz4*z36;

 fyz16=(l2-lyz1)*k(21);
 fyz16x=fyz16/lyz1*x16;
 fyz16y=fyz16/lyz1*y16;
 fyz16z=fyz16/lyz1*z16;
 myz16x=L0(21)/lyz1*x16;
 myz16y=L0(21)/lyz1*y16;
 myz16z=L0(21)/lyz1*z16;

 fyz25=(l2-lyz2)*k(22);
 fyz25x=fyz25/lyz2*x25;
 fyz25y=fyz25/lyz2*y25;
 fyz25z=fyz25/lyz2*z25;
 myz25x=L0(22)/lyz2*x25;
 myz25y=L0(22)/lyz2*y25;
 myz25z=L0(22)/lyz2*z25;

 fyz38=(l2-lyz3)*k(23);
 fyz38x=fyz38/lyz3*x38;
 fyz38y=fyz38/lyz3*y38;
 fyz38z=fyz38/lyz3*z38;
 myz38x=L0(23)/lyz3*x38;
 myz38y=L0(23)/lyz3*y38;
 myz38z=L0(23)/lyz3*z38;

 fyz47=(l2-lyz4)*k(24);
 fyz47x=fyz47/lyz4*x47;
 fyz47y=fyz47/lyz4*y47;
 fyz47z=fyz47/lyz4*z47;
 myz47x=L0(24)/lyz4*x47;
 myz47y=L0(24)/lyz4*y47;
 myz47z=L0(24)/lyz4*z47;
 % getdirection of spring xyz
 lxyz1=(sum((v(1,:)-v(7,:)).^2))^0.5;
 x17=(v(1,1)-v(7,1));
 y17=(v(1,2)-v(7,2));
 z17=(v(1,3)-v(7,3));
 lxyz2=(sum((v(3,:)-v(5,:)).^2))^0.5;
 x35=(v(3,1)-v(5,1));
 y35=(v(3,2)-v(5,2));
 z35=(v(3,3)-v(5,3));
 lxyz3=(sum((v(2,:)-v(8,:)).^2))^0.5;
 x28=(v(2,1)-v(8,1));
 y28=(v(2,2)-v(8,2));
 z28=(v(2,3)-v(8,3));
 lxyz4=(sum((v(4,:)-v(6,:)).^2))^0.5;
 x46=(v(4,1)-v(6,1));
 y46=(v(4,2)-v(6,2));
 z46=(v(4,3)-v(6,3));
% get force and sine motion of spring
 fxyz17=(l3-lxyz1)*k(25);
 fxyz17x=fxyz17/lxyz1*x17;
 fxyz17y=fxyz17/lxyz1*y17;
 fxyz17z=fxyz17/lxyz1*z17;
 mxyz17x=L0(25)/lxyz1*x17;
 mxyz17y=L0(25)/lxyz1*y17;
 mxyz17z=L0(25)/lxyz1*z17;

 fxyz35=(l3-lxyz2)*k(26);
 fxyz35x=fxyz35/lxyz2*x35;
 fxyz35y=fxyz35/lxyz2*y35;
 fxyz35z=fxyz35/lxyz2*z35;
 mxyz35x=L0(26)/lxyz2*x35;
 mxyz35y=L0(26)/lxyz2*y35;
 mxyz35z=L0(26)/lxyz2*z35;

 fxyz28=(l3-lxyz3)*k(27);
 fxyz28x=fxyz28/lxyz3*x28;
 fxyz28y=fxyz28/lxyz3*y28;
 fxyz28z=fxyz28/lxyz3*z28;
 mxyz28x=L0(27)/lxyz3*x28;
 mxyz28y=L0(27)/lxyz3*y28;
 mxyz28z=L0(27)/lxyz3*z28;

 fxyz46=(l3-lxyz4)*k(28);
 fxyz46x=fxyz46/lxyz4*x46;
 fxyz46y=fxyz46/lxyz4*y46;
 fxyz46z=fxyz46/lxyz4*z46;
 mxyz46x=L0(28)/lxyz4*x46;
 mxyz46y=L0(28)/lxyz4*y46;
 mxyz46z=L0(28)/lxyz4*z46;
 %% put commbine force and motion into 8 mass point in xyz direction

 Fspring=zeros(8,3);
 Fspring(1,1)=(fx14x+fy12x+fz15x+fxy13x+fxz18x+fyz16x+fxyz17x);
 Fspring(1,2)=(fx14y+fy12y+fz15y+fxy13y+fxz18y+fyz16y+fxyz17y);
 Fspring(1,3)=(fx14z+fy12z+fz15z+fxy13z+fxz18z+fyz16z+fxyz17z);

 Fspring(2,1)=(fx23x-fy12x+fz26x+fxy24x+fxz27x+fyz25x+fxyz28x);
 Fspring(2,2)=(fx23y-fy12y+fz26y+fxy24y+fxz27y+fyz25y+fxyz28y);
 Fspring(2,3)=(fx23z-fy12z+fz26z+fxy24z+fxz27z+fyz25z+fxyz28z);

 Fspring(3,1)=(-fx23x+fy34x+fz37x-fxy13x+fxz36x+fyz38x+fxyz35x);
 Fspring(3,2)=(-fx23y+fy34y+fz37y-fxy13y+fxz36y+fyz38y+fxyz35y);
 Fspring(3,3)=(-fx23z+fy34z+fz37z-fxy13z+fxz36z+fyz38z+fxyz35z);

 Fspring(4,1)=(-fx14x-fy34x+fz48x-fxy24x+fxz45x+fyz47x+fxyz46x);
 Fspring(4,2)=(-fx14y-fy34y+fz48y-fxy24y+fxz45y+fyz47y+fxyz46y);
 Fspring(4,3)=(-fx14z-fy34z+fz48z-fxy24z+fxz45z+fyz47z+fxyz46z);

 Fspring(5,1)=(fx58x+fy56x-fz15x+fxy57x-fxz45x-fyz25x-fxyz35x);
 Fspring(5,2)=(fx58y+fy56y-fz15y+fxy57y-fxz45y-fyz25y-fxyz35y);
 Fspring(5,3)=(fx58z+fy56z-fz15z+fxy57z-fxz45z-fyz25z-fxyz35z);

 Fspring(6,1)=(fx67x-fy56x-fz26x+fxy68x-fxz36x-fyz16x-fxyz46x);
 Fspring(6,2)=(fx67y-fy56y-fz26y+fxy68y-fxz36y-fyz16y-fxyz46y);
 Fspring(6,3)=(fx67z-fy56z-fz26z+fxy68z-fxz36z-fyz16z-fxyz46z);

 Fspring(7,1)=(-fx67x+fy78x-fz37x-fxy57x-fxz27x-fyz47x-fxyz17x);
 Fspring(7,2)=(-fx67y+fy78y-fz37y-fxy57y-fxz27y-fyz47y-fxyz17y);
 Fspring(7,3)=(-fx67z+fy78z-fz37z-fxy57z-fxz27z-fyz47z-fxyz17z);

 Fspring(8,1)=(-fx58x-fy78x-fz48x-fxy68x-fxz18x-fyz38x-fxyz28x);
 Fspring(8,2)=(-fx58y-fy78y-fz48y-fxy68y-fxz18y-fyz38y-fxyz28y);
 Fspring(8,3)=(-fx58z-fy78z-fz48z-fxy68z-fxz18z-fyz38z-fxyz28z);

 moveL0=zeros(8,3);
 moveL0(1,1)=(mx14x+my12x+mz15x+mxy13x+mxz18x+myz16x+mxyz17x)/7;
 moveL0(1,2)=(mx14y+my12y+mz15y+mxy13y+mxz18y+myz16y+mxyz17y)/7;
 moveL0(1,3)=(mx14z+my12z+mz15z+mxy13z+mxz18z+myz16z+mxyz17z)/7;

 moveL0(2,1)=(mx23x-my12x+mz26x+mxy24x+mxz27x+myz25x+mxyz28x)/7;
 moveL0(2,2)=(mx23y-my12y+mz26y+mxy24y+mxz27y+myz25y+mxyz28y)/7;
 moveL0(2,3)=(mx23z-my12z+mz26z+mxy24z+mxz27z+myz25z+mxyz28z)/7;

 moveL0(3,1)=(-mx23x+my34x+mz37x-mxy13x+mxz36x+myz38x+mxyz35x)/7;
 moveL0(3,2)=(-mx23y+my34y+mz37y-mxy13y+mxz36y+myz38y+mxyz35y)/7;
 moveL0(3,3)=(-mx23z+my34z+mz37z-mxy13z+mxz36z+myz38z+mxyz35z)/7;

 moveL0(4,1)=(-mx14x-my34x+mz48x-mxy24x+mxz45x+myz47x+mxyz46x)/7;
 moveL0(4,2)=(-mx14y-my34y+mz48y-mxy24y+mxz45y+myz47y+mxyz46y)/7;
 moveL0(4,3)=(-mx14z-my34z+mz48z-mxy24z+mxz45z+myz47z+mxyz46z)/7;

 moveL0(5,1)=(mx58x+my56x-mz15x+mxy57x-mxz45x-myz25x-mxyz35x)/7;
 moveL0(5,2)=(mx58y+my56y-mz15y+mxy57y-mxz45y-myz25y-mxyz35y)/7;
 moveL0(5,3)=(mx58z+my56z-mz15z+mxy57z-mxz45z-myz25z-mxyz35z)/7;

 moveL0(6,1)=(mx67x-my56x-mz26x+mxy68x-mxz36x-myz16x-mxyz46x)/7;
 moveL0(6,2)=(mx67y-my56y-mz26y+mxy68y-mxz36y-myz16y-mxyz46y)/7;
 moveL0(6,3)=(mx67z-my56z-mz26z+mxy68z-mxz36z-myz16z-mxyz46z)/7;

 moveL0(7,1)=(-mx67x+my78x-mz37x-mxy57x-mxz27x-myz47x-mxyz17x)/7;
 moveL0(7,2)=(-mx67y+my78y-mz37y-mxy57y-mxz27y-myz47y-mxyz17y)/7;
 moveL0(7,3)=(-mx67z+my78z-mz37z-mxy57z-mxz27z-myz47z-mxyz17z)/7;

 moveL0(8,1)=(-mx58x-my78x-mz48x-mxy68x-mxz18x-myz38x-mxyz28x)/7;
 moveL0(8,2)=(-mx58y-my78y-mz48y-mxy68y-mxz18y-myz38y-mxyz28y)/7;
 moveL0(8,3)=(-mx58z-my78z-mz48z-mxy68z-mxz18z-myz38z-mxyz28z)/7;
end

 
 
 
 