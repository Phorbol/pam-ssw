
/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp:     file format elf64-x86-64


Disassembly of section .text:

00000000005bf699 <ssw_fixlat_mp_ssw_move_+0x2f39>:
  5bf699:	f2 0f 10 8b 28 db 02 	movsd  xmm1,QWORD PTR [rbx+0x2db28]
  5bf6a0:	00
  5bf6a1:	66 0f 2f c8          	comisd xmm1,xmm0
  5bf6a5:	77 0d                	ja     5bf6b4 <.debug_info_seg+0xfe2>
  5bf6a7:	44 3b bb d0 da 02 00 	cmp    r15d,DWORD PTR [rbx+0x2dad0]
  5bf6ae:	0f 8c f6 f1 ff ff    	jl     5be8aa <.debug_info_seg+0x1d8>
  5bf6b4:	66 0f ef d2          	pxor   xmm2,xmm2
  5bf6b8:	f2 41 0f 2a 16       	cvtsi2sd xmm2,DWORD PTR [r14]
  5bf6bd:	f2 0f 10 0d d3 b7 32 	movsd  xmm1,QWORD PTR [rip+0x732b7d3]        # 78eae98 <ssw_fixlat_mp_ssw_move_$ENE_BEFORE.0.7>
  5bf6c4:	07
  5bf6c5:	f2 0f 10 85 b8 fc ff 	movsd  xmm0,QWORD PTR [rbp-0x348]
  5bf6cc:	ff
  5bf6cd:	48 8d 05 7c 3c b4 1c 	lea    rax,[rip+0x1cb43c7c]        # 1d103350 <pot_bond_var_def_mp_biasperatom_save_>
  5bf6d4:	f2 0f 5c c1          	subsd  xmm0,xmm1
  5bf6d8:	f2 0f 5e c2          	divsd  xmm0,xmm2
  5bf6dc:	f2 0f 59 05 1c 67 48 	mulsd  xmm0,QWORD PTR [rip+0x448671c]        # 4a45e00 <__STRLITPACK_643.0.30+0xb8>
  5bf6e3:	04
  5bf6e4:	f2 0f 11 8d a8 fc ff 	movsd  QWORD PTR [rbp-0x358],xmm1
  5bf6eb:	ff
  5bf6ec:	f2 0f 11 85 a0 fc ff 	movsd  QWORD PTR [rbp-0x360],xmm0
  5bf6f3:	ff
  5bf6f4:	f2 0f 11 00          	movsd  QWORD PTR [rax],xmm0
