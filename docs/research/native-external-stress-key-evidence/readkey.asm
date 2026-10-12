
/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp:     file format elf64-x86-64


Disassembly of section .text:

00000000004113fc <readkey_+0x323c>:
  4113fc:	48 89 df             	mov    %rbx,%rdi
  4113ff:	be 64 00 00 00       	mov    $0x64,%esi
  411404:	48 ba b4 ff a2 04 00 	movabs $0x4a2ffb4,%rdx
  41140b:	00 00 00 
  41140e:	b9 08 00 00 00       	mov    $0x8,%ecx
  411413:	41 b8 02 00 00 00    	mov    $0x2,%r8d
  411419:	41 ff d7             	call   *%r15
  41141c:	85 c0                	test   %eax,%eax
  41141e:	74 2a                	je     41144a <.debug_info_seg+0x377>
  411420:	48 bf 44 0b a3 04 00 	movabs $0x4a30b44,%rdi
  411427:	00 00 00 
  41142a:	48 be 18 1d d7 05 00 	movabs $0x5d71d18,%rsi
  411431:	00 00 00 
  411434:	ba 10 00 00 00       	mov    $0x10,%edx
  411439:	48 b9 30 c7 47 00 00 	movabs $0x47c730,%rcx
  411440:	00 00 00 
  411443:	ff d1                	call   *%rcx
  411445:	e9 1d 07 00 00       	jmp    411b67 <.debug_info_seg+0xa94>
