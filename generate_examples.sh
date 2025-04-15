#! /bin/bash

path='/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring'
iter=100

# command
execute_cmd='/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -javaagent:/opt/idea/lib/idea_rt.jar=38341:/opt/idea/bin -Dfile.encoding=UTF-8 -classpath /usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/charsets.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/cldrdata.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/dnsns.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/icedtea-sound.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/jaccess.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/java-atk-wrapper.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/localedata.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/nashorn.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/sunec.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/sunjce_provider.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/sunpkcs11.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/ext/zipfs.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/jce.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/jsse.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/management-agent.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/resources.jar:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/rt.jar:/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring/doupr-execution/target/classes:/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring/doupr-optimization/target/classes:/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring/doupr-core/target/classes:/home/fakher/.m2/repository/com/opencsv/opencsv/4.0/opencsv-4.0.jar:/home/fakher/.m2/repository/commons-beanutils/commons-beanutils/1.9.3/commons-beanutils-1.9.3.jar:/home/fakher/.m2/repository/commons-logging/commons-logging/1.2/commons-logging-1.2.jar:/home/fakher/.m2/repository/commons-collections/commons-collections/3.2.2/commons-collections-3.2.2.jar:/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring/doupr-evaluation/target/classes:/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring/doupr-export/target/classes:/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring/doupr-gui/target/classes:/home/fakher/.m2/repository/org/knowm/xchart/xchart/3.5.2/xchart-3.5.2.jar:/home/fakher/.m2/repository/de/erichseifert/vectorgraphics2d/VectorGraphics2D/0.13/VectorGraphics2D-0.13.jar:/home/fakher/.m2/repository/org/knowm/xchart/xchart-demo/3.5.2/xchart-demo-3.5.2.jar:/home/fakher/.m2/repository/org/jfree/jfreechart/1.0.19/jfreechart-1.0.19.jar:/home/fakher/.m2/repository/org/jfree/jcommon/1.0.23/jcommon-1.0.23.jar:/home/fakher/.m2/repository/org/swinglabs/swingx/1.6.1/swingx-1.6.1.jar:/home/fakher/.m2/repository/com/jhlabs/filters/2.0.235/filters-2.0.235.jar:/home/fakher/.m2/repository/org/swinglabs/swing-worker/1.1/swing-worker-1.1.jar:/home/fakher/.m2/repository/args4j/args4j/2.33/args4j-2.33.jar:/home/fakher/.m2/repository/org/slf4j/slf4j-api/1.8.0-beta2/slf4j-api-1.8.0-beta2.jar:/home/fakher/.m2/repository/com/google/guava/guava/23.6-jre/guava-23.6-jre.jar:/home/fakher/.m2/repository/com/google/code/findbugs/jsr305/1.3.9/jsr305-1.3.9.jar:/home/fakher/.m2/repository/org/checkerframework/checker-compat-qual/2.0.0/checker-compat-qual-2.0.0.jar:/home/fakher/.m2/repository/com/google/errorprone/error_prone_annotations/2.1.3/error_prone_annotations-2.1.3.jar:/home/fakher/.m2/repository/com/google/j2objc/j2objc-annotations/1.1/j2objc-annotations-1.1.jar:/home/fakher/.m2/repository/org/codehaus/mojo/animal-sniffer-annotations/1.14/animal-sniffer-annotations-1.14.jar:/home/fakher/.m2/repository/org/apache/httpcomponents/httpcore/4.4.8/httpcore-4.4.8.jar:/home/fakher/.m2/repository/org/uma/jmetal/jmetal-core/5.2/jmetal-core-5.2.jar:/home/fakher/.m2/repository/org/apache/maven/reporting/maven-reporting-api/3.0/maven-reporting-api-3.0.jar:/home/fakher/.m2/repository/org/apache/maven/doxia/doxia-sink-api/1.0/doxia-sink-api-1.0.jar:/home/fakher/.m2/repository/org/apache/commons/commons-lang3/3.7/commons-lang3-3.7.jar:/home/fakher/.m2/repository/org/apache/commons/commons-math3/3.3/commons-math3-3.3.jar:/home/fakher/.m2/repository/commons-io/commons-io/2.4/commons-io-2.4.jar:/home/fakher/.m2/repository/org/hamcrest/hamcrest-all/1.3/hamcrest-all-1.3.jar:/home/fakher/.m2/repository/org/uma/jmetal/jmetal-algorithm/5.2/jmetal-algorithm-5.2.jar:/home/fakher/.m2/repository/org/uma/jmetal/jmetal-exec/5.2/jmetal-exec-5.2.jar:/home/fakher/.m2/repository/org/uma/jmetal/jmetal-problem/5.2/jmetal-problem-5.2.jar:/home/fakher/.m2/repository/org/mockito/mockito-core/2.7.19/mockito-core-2.7.19.jar:/home/fakher/.m2/repository/net/bytebuddy/byte-buddy/1.7.11/byte-buddy-1.7.11.jar:/home/fakher/.m2/repository/net/bytebuddy/byte-buddy-agent/1.7.11/byte-buddy-agent-1.7.11.jar:/home/fakher/.m2/repository/org/objenesis/objenesis/2.5/objenesis-2.5.jar:/home/fakher/.m2/repository/log4j/log4j/1.2.17/log4j-1.2.17.jar:/home/fakher/.m2/repository/joda-time/joda-time/2.9.9/joda-time-2.9.9.jar:/home/fakher/.m2/repository/org/apache/commons/commons-text/1.1/commons-text-1.1.jar thiagodnf.doupr.execution.ExploreConsole'

# params
crossoverProbability=( 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.5) 
mutationProbability=( 0.5 0.5 0.5 0.6 0.6 0.6 0.7 0.5 0.6)
blocks_file='/home/fakher/Desktop/sbse/VA_Refactoring_Tool_Web_UI/sbse-Refactoring/doupr-execution/src/main/resources/jhotdraw-7.6/jhotdraw.blocks'


for xprob in ${crossoverProbability[@]}
	do
		for mprob in ${mutationProbability[@]}
			do
				# execute code 
				$execute_cmd -in $blocks_file -cProb $xprob -mProb $mprob;
				echo "optimizer executed"
				
				# move to data files
				mv "${path}/data/last release/executiontraces.txt" "${path}/data/last release/data/executiontraces-${iter}.txt";
				
				# clean output files
				rm output*-var.txt
				rm -r ./output
				let iter=$iter+1;
			done
	done	


