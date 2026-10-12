
/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp:     file format elf64-x86-64


Disassembly of section .text:

0000000000414d40 <module_pes_reset_>:
  414d40:	55                   	push   %rbp
  414d41:	48 89 e5             	mov    %rsp,%rbp
  414d44:	41 54                	push   %r12
  414d46:	41 55                	push   %r13
  414d48:	41 57                	push   %r15
  414d4a:	53                   	push   %rbx
  414d4b:	49 b9 20 28 3e 05 00 	movabs $0x53e2820,%r9
  414d52:	00 00 00 
  414d55:	49 ba 18 28 3e 05 00 	movabs $0x53e2818,%r10
  414d5c:	00 00 00 
  414d5f:	49 bf 10 28 3e 05 00 	movabs $0x53e2810,%r15
  414d66:	00 00 00 
  414d69:	48 b8 00 00 00 00 00 	movabs $0x3ff0000000000000,%rax
  414d70:	00 f0 3f 
  414d73:	48 ba 08 28 3e 05 00 	movabs $0x53e2808,%rdx
  414d7a:	00 00 00 
  414d7d:	48 be 00 28 3e 05 00 	movabs $0x53e2800,%rsi
  414d84:	00 00 00 
  414d87:	48 bf f8 27 3e 05 00 	movabs $0x53e27f8,%rdi
  414d8e:	00 00 00 
  414d91:	49 89 01             	mov    %rax,(%r9)
  414d94:	49 b9 e8 27 3e 05 00 	movabs $0x53e27e8,%r9
  414d9b:	00 00 00 
  414d9e:	48 bb 00 00 00 00 00 	movabs $0x4000000000000000,%rbx
  414da5:	00 00 40 
  414da8:	49 bb 00 00 00 00 00 	movabs $0x4004000000000000,%r11
  414daf:	00 04 40 
  414db2:	49 89 1a             	mov    %rbx,(%r10)
  414db5:	49 ba 30 1d d7 05 00 	movabs $0x5d71d30,%r10
  414dbc:	00 00 00 
  414dbf:	4d 89 1f             	mov    %r11,(%r15)
  414dc2:	49 bb 28 1d d7 05 00 	movabs $0x5d71d28,%r11
  414dc9:	00 00 00 
  414dcc:	48 b8 b8 1e 85 eb 51 	movabs $0x3f9eb851eb851eb8,%rax
  414dd3:	b8 9e 3f 
  414dd6:	48 b9 00 00 00 00 00 	movabs $0x403e000000000000,%rcx
  414ddd:	00 3e 40 
  414de0:	41 b0 20             	mov    $0x20,%r8b
  414de3:	49 bd e0 78 9a 04 00 	movabs $0x49a78e0,%r13
  414dea:	00 00 00 
  414ded:	48 89 02             	mov    %rax,(%rdx)
  414df0:	48 ba 14 11 a3 04 00 	movabs $0x4a31114,%rdx
  414df7:	00 00 00 
  414dfa:	48 89 0e             	mov    %rcx,(%rsi)
  414dfd:	be 64 00 00 00       	mov    $0x64,%esi
  414e02:	c7 07 ff ff ff ff    	movl   $0xffffffff,(%rdi)
  414e08:	48 bf 80 27 3e 05 00 	movabs $0x53e2780,%rdi
  414e0f:	00 00 00 
  414e12:	45 88 41 03          	mov    %r8b,0x3(%r9)
  414e16:	b9 10 00 00 00       	mov    $0x10,%ecx
  414e1b:	45 88 41 04          	mov    %r8b,0x4(%r9)
  414e1f:	45 33 ff             	xor    %r15d,%r15d
  414e22:	45 88 41 05          	mov    %r8b,0x5(%r9)
  414e26:	45 88 41 06          	mov    %r8b,0x6(%r9)
  414e2a:	45 88 41 07          	mov    %r8b,0x7(%r9)
  414e2e:	45 88 41 08          	mov    %r8b,0x8(%r9)
  414e32:	45 88 41 09          	mov    %r8b,0x9(%r9)
  414e36:	45 33 c0             	xor    %r8d,%r8d
  414e39:	41 c6 01 70          	movb   $0x70,(%r9)
  414e3d:	41 c6 41 01 62       	movb   $0x62,0x1(%r9)
  414e42:	41 c6 41 02 65       	movb   $0x65,0x2(%r9)
  414e47:	45 89 3a             	mov    %r15d,(%r10)
  414e4a:	45 89 3b             	mov    %r15d,(%r11)
  414e4d:	41 ff d5             	call   *%r13
  414e50:	49 bc f8 1c d7 05 00 	movabs $0x5d71cf8,%r12
  414e57:	00 00 00 
  414e5a:	49 b9 20 1d d7 05 00 	movabs $0x5d71d20,%r9
  414e61:	00 00 00 
  414e64:	48 b8 18 1d d7 05 00 	movabs $0x5d71d18,%rax
  414e6b:	00 00 00 
  414e6e:	48 ba 10 1d d7 05 00 	movabs $0x5d71d10,%rdx
  414e75:	00 00 00 
  414e78:	48 b9 70 27 3e 05 00 	movabs $0x53e2770,%rcx
  414e7f:	00 00 00 
  414e82:	48 bf 00 1d d7 05 00 	movabs $0x5d71d00,%rdi
  414e89:	00 00 00 
  414e8c:	49 b8 f0 1c d7 05 00 	movabs $0x5d71cf0,%r8
  414e93:	00 00 00 
  414e96:	49 bb 78 27 3e 05 00 	movabs $0x53e2778,%r11
  414e9d:	00 00 00 
  414ea0:	45 89 3c 24          	mov    %r15d,(%r12)
  414ea4:	45 33 e4             	xor    %r12d,%r12d
  414ea7:	4d 89 21             	mov    %r12,(%r9)
  414eaa:	49 b9 68 27 3e 05 00 	movabs $0x53e2768,%r9
  414eb1:	00 00 00 
  414eb4:	44 89 38             	mov    %r15d,(%rax)
