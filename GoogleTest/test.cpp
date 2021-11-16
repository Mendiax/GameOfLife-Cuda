#include "pch.h"
#include <iostream>

TEST(TestCaseName, TestName) {
	std::cout << "OK" << std::endl;
	EXPECT_EQ(1, 1);
	EXPECT_TRUE(true);
}