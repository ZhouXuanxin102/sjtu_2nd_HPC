#include <iostream>
#include <vector>
#include "main.hpp"


void find_block1_wang(uint32 block[], const uint32 IV[]);
void find_block1_stevens_11(uint32 block[], const uint32 IV[]);
void find_block1_stevens_10(uint32 block[], const uint32 IV[]);
void find_block1_stevens_01(uint32 block[], const uint32 IV[]);
void find_block1_stevens_00(uint32 block[], const uint32 IV[]);

void find_block1(uint32 block[], const uint32 IV[])
{
	if ( ((IV[1]^IV[2])&(1<<31))==0 && ((IV[1]^IV[3])&(1<<31))==0
		&& (IV[3]&(1<<25))==0 && (IV[2]&(1<<25))==0 && (IV[1]&(1<<25))==0 && ((IV[2]^IV[1])&1)==0 
	   )
	{
		uint32 IV2[4] = { IV[0]+(1<<31), IV[1]+(1<<31)+(1<<25), IV[2]+(1<<31)+(1<<25), IV[3]+(1<<31)+(1<<25) };
		if ((IV[1]&(1<<6))!=0 && (IV[1]&1)!=0) {
			find_block1_stevens_11(block, IV2);
		} else if ((IV[1]&(1<<6))!=0 && (IV[1]&1)==0) {
			find_block1_stevens_10(block, IV2);
		} else if ((IV[1]&(1<<6))==0 && (IV[1]&1)!=0) {
			find_block1_stevens_01(block, IV2);
		} else {
			find_block1_stevens_00(block, IV2);
		}
		block[4] += 1<<31;
		block[11] += 1<<15;
		block[14] += 1<<31;
	} else {
		find_block1_wang(block, IV);
	}
}
