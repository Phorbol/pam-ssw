
/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp:     file format elf64-x86-64


Disassembly of section .text:

00000000005e80d0 <ssw_crystal_basic_mp_get_random_mode0_>:
  5e80d0:	55                   	push   rbp
  5e80d1:	48 89 e5             	mov    rbp,rsp
  5e80d4:	41 57                	push   r15
  5e80d6:	53                   	push   rbx
  5e80d7:	48 81 ec d0 00 00 00 	sub    rsp,0xd0
  5e80de:	49 89 ff             	mov    r15,rdi
  5e80e1:	48 8d 1d b8 56 e0 04 	lea    rbx,[rip+0x4e056b8]        # 53ed7a0 <ssw_parameters_mp_para_>
  5e80e8:	0f 10 05 11 af ee 04 	movups xmm0,XMMWORD PTR [rip+0x4eeaf11]        # 54d3000 <ssw_crystal_basic_mp_get_random_mode0_$blk.var$3523>
  5e80ef:	0f 10 0d 1a af ee 04 	movups xmm1,XMMWORD PTR [rip+0x4eeaf1a]        # 54d3010 <ssw_crystal_basic_mp_get_random_mode0_$blk.var$3523+0x10>
  5e80f6:	0f 10 15 23 af ee 04 	movups xmm2,XMMWORD PTR [rip+0x4eeaf23]        # 54d3020 <ssw_crystal_basic_mp_get_random_mode0_$blk.var$3523+0x20>
  5e80fd:	0f 10 1d 2c af ee 04 	movups xmm3,XMMWORD PTR [rip+0x4eeaf2c]        # 54d3030 <ssw_crystal_basic_mp_get_random_mode0_$blk.var$3523+0x30>
  5e8104:	0f 10 25 35 af ee 04 	movups xmm4,XMMWORD PTR [rip+0x4eeaf35]        # 54d3040 <ssw_crystal_basic_mp_get_random_mode0_$blk.var$3523+0x40>
  5e810b:	48 8b 05 3e af ee 04 	mov    rax,QWORD PTR [rip+0x4eeaf3e]        # 54d3050 <ssw_crystal_basic_mp_get_random_mode0_$blk.var$3523+0x50>
  5e8112:	8b 8b 58 db 02 00    	mov    ecx,DWORD PTR [rbx+0x2db58]
  5e8118:	0f 11 45 90          	movups XMMWORD PTR [rbp-0x70],xmm0
  5e811c:	0f 11 4d a0          	movups XMMWORD PTR [rbp-0x60],xmm1
  5e8120:	0f 11 55 b0          	movups XMMWORD PTR [rbp-0x50],xmm2
  5e8124:	0f 11 5d c0          	movups XMMWORD PTR [rbp-0x40],xmm3
  5e8128:	0f 11 65 d0          	movups XMMWORD PTR [rbp-0x30],xmm4
  5e812c:	48 89 45 e0          	mov    QWORD PTR [rbp-0x20],rax
  5e8130:	49 8b 37             	mov    rsi,QWORD PTR [r15]
  5e8133:	85 c9                	test   ecx,ecx
  5e8135:	75 0e                	jne    5e8145 <ssw_crystal_basic_mp_get_random_mode0_+0x75>
  5e8137:	32 c0                	xor    al,al
  5e8139:	c7 86 60 22 00 00 00 	mov    DWORD PTR [rsi+0x2260],0x0
  5e8140:	00 00 00
  5e8143:	eb 2e                	jmp    5e8173 <ssw_crystal_basic_mp_get_random_mode0_+0xa3>
  5e8145:	0f 8e 92 01 00 00    	jle    5e82dd <ssw_crystal_basic_mp_get_random_mode0_+0x20d>
  5e814b:	8b 86 74 2a 00 00    	mov    eax,DWORD PTR [rsi+0x2a74]
  5e8151:	99                   	cdq
  5e8152:	f7 f9                	idiv   ecx
  5e8154:	83 fa 01             	cmp    edx,0x1
  5e8157:	74 0e                	je     5e8167 <ssw_crystal_basic_mp_get_random_mode0_+0x97>
  5e8159:	c7 86 60 22 00 00 00 	mov    DWORD PTR [rsi+0x2260],0x0
  5e8160:	00 00 00
  5e8163:	32 c0                	xor    al,al
  5e8165:	eb 0c                	jmp    5e8173 <ssw_crystal_basic_mp_get_random_mode0_+0xa3>
  5e8167:	c7 86 60 22 00 00 ff 	mov    DWORD PTR [rsi+0x2260],0xffffffff
  5e816e:	ff ff ff
  5e8171:	b0 ff                	mov    al,0xff
  5e8173:	a8 01                	test   al,0x1
  5e8175:	74 13                	je     5e818a <ssw_crystal_basic_mp_get_random_mode0_+0xba>
  5e8177:	49 8b 47 38          	mov    rax,QWORD PTR [r15+0x38]
  5e817b:	4c 89 ff             	mov    rdi,r15
  5e817e:	48 8d 75 90          	lea    rsi,[rbp-0x70]
  5e8182:	ff 90 10 02 00 00    	call   QWORD PTR [rax+0x210]
  5e8188:	eb 11                	jmp    5e819b <ssw_crystal_basic_mp_get_random_mode0_+0xcb>
  5e818a:	49 8b 47 38          	mov    rax,QWORD PTR [r15+0x38]
  5e818e:	4c 89 ff             	mov    rdi,r15
  5e8191:	48 8d 75 90          	lea    rsi,[rbp-0x70]
  5e8195:	ff 90 18 02 00 00    	call   QWORD PTR [rax+0x218]
