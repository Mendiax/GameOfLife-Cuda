#include "pch.h"
#include <engine/cuda/kernel.cuh>
#include <iostream>

TEST(TestCaseName, TestName) {
	mainCuda();
	std::cout << "OK" << std::endl;
	EXPECT_EQ(1, 1);
	EXPECT_TRUE(true);
}