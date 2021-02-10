#! /bin/sh

# functional tests for Ripser++ using shunit2
# after installation, in the build folder type: `make tests` to run the following functional tests

testCelegans() {
	assertSame 'error on celegans barcode computation' "echo $(cat ./testing/ripser++_testing_expected_barcodes/celegans.gpu.barcodes)" "echo $(./ripser++ ../examples/celegans.distance_matrix 2> /dev/null)"
}
testDragon1000() {
	assertSame 'error on dragon1000 barcode computation' "echo $(cat ./testing/ripser++_testing_expected_barcodes/dragon1000.gpu.barcodes)" "echo $(./ripser++ ../examples/dragon1000.distance_matrix 2> /dev/null)"
}
testHIV() {
	assertSame 'error on HIV barcode computation' "echo $(cat ./testing/ripser++_testing_expected_barcodes/HIV.gpu.barcodes)" "echo $(./ripser++ ../examples/HIV.distance_matrix 2> /dev/null)"
}
testo3_4096() {
	assertSame 'error on o3_4096 barcode computation' "echo $(cat ./testing/ripser++_testing_expected_barcodes/o3_4096.gpu.barcodes)" "echo $(./ripser++ --threshold 1.4 --sparse --format point-cloud ../examples/o3_4096.point_cloud 2> /dev/null)"
}
testSphere_3_192() {
	assertSame 'error on sphere_3_192 barcode computation' "echo $(cat ./testing/ripser++_testing_expected_barcodes/sphere_3_192.gpu.barcodes)" "echo $(./ripser++ ../examples/sphere_3_192.distance_matrix.lower_triangular 2> /dev/null)"
}
testVicsek300() {
    assertSame 'error on Vicsek300 barcode computation' "echo $(cat ./testing/ripser++_testing_expected_barcodes/Vicsek300_300_of_300.gpu.barcodes)" "echo $(./ripser++ ../examples/Vicsek300_300_of_300.distance_matrix 2> /dev/null)"
}

# Load shUnit2.
. ./testing/shunit2/shunit2