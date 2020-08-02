package com.example.sanadistancia;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Environment;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;
import java.lang.Math;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    boolean startYolo = false;
    boolean firstTimeYolo = false;
    Net tinyYolo;

    public void YOLO(View Button){

        if (startYolo == false){




            startYolo = true;

            if (firstTimeYolo == false){


                firstTimeYolo = true;
                String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg"  ;
                String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";

                tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);


              //






            }



        }

        else{

            startYolo = false;


        }




    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);


        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:

                        super.onManagerConnected(status);
                        break;
                }


            }

        };
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba();

        if (startYolo == true) {

            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            int H = frame.rows();
            int W = frame.cols();
            int FW = W;
            if(W<1075)
            {
                FW = 1075;
            }
            Mat FR = new Mat(H+210,FW, CvType.CV_8UC3,new Scalar(255,255,255));
            int FH = H+210;
            Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416,416),new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);


            tinyYolo.setInput(imageBlob);



            java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

            List<String> outBlobNames = new java.util.ArrayList<>();
            outBlobNames.add(0, "yolo_16");
            outBlobNames.add(1, "yolo_23");

            tinyYolo.forward(result,outBlobNames);


            float confThreshold = 0.3f;


            List<String> cocoNames = Arrays.asList("a person", "a bicycle", "a motorbike", "an airplane", "a bus", "a train", "a truck", "a boat", "a traffic light", "a fire hydrant", "a stop sign", "a parking meter", "a car", "a bench", "a bird", "a cat", "a dog", "a horse", "a sheep", "a cow", "an elephant", "a bear", "a zebra", "a giraffe", "a backpack", "an umbrella", "a handbag", "a tie", "a suitcase", "a frisbee", "skis", "a snowboard", "a sports ball", "a kite", "a baseball bat", "a baseball glove", "a skateboard", "a surfboard", "a tennis racket", "a bottle", "a wine glass", "a cup", "a fork", "a knife", "a spoon", "a bowl", "a banana", "an apple", "a sandwich", "an orange", "broccoli", "a carrot", "a hot dog", "a pizza", "a doughnut", "a cake", "a chair", "a sofa", "a potted plant", "a bed", "a dining table", "a toilet", "a TV monitor", "a laptop", "a computer mouse", "a remote control", "a keyboard", "a cell phone", "a microwave", "an oven", "a toaster", "a sink", "a refrigerator", "a book", "a clock", "a vase", "a pair of scissors", "a teddy bear", "a hair drier", "a toothbrush");
            List<Integer> clsIds = new ArrayList<>();
            List<Integer> CentrosX = new ArrayList<>();
            List<Integer> CentrosY = new ArrayList<>();
            List<Integer> Width_Arr = new ArrayList<>();
            List<Integer> Height_Arr = new ArrayList<>();
            List<Integer> status = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect> rects = new ArrayList<>();




            for (int i = 0; i < result.size(); ++i)
            {

                Mat level = result.get(i);

                for (int j = 0; j < level.rows(); ++j)
                {
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());

                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);




                    float confidence = (float)mm.maxVal;


                    Point classIdPoint = mm.maxLoc;



                    if (confidence > confThreshold)
                    {
                        if((cocoNames.get((int)classIdPoint.x)) == "a person")
                        {
                            int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                            int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                            int width = (int) (row.get(0, 2)[0] * frame.cols());
                            int height = (int) (row.get(0, 3)[0] * frame.rows());


                            int left = centerX - width / 2;
                            int top = centerY - height / 2;

                            clsIds.add((int) classIdPoint.x);
                            confs.add((float) confidence);
                            CentrosX.add((int) centerX);
                            CentrosY.add((int) centerY);
                            Width_Arr.add((int) width);
                            Height_Arr.add((int) height);
                            status.add((int) 0 );


                            rects.add(new Rect(left, top, width, height));
                        }
                    }
                }
            }
            int ArrayLength = confs.size();

            if (ArrayLength>=1) {
                // Apply non-maximum suppression procedure.
                float nmsThresh = 0.2f;

                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

                Rect[] boxesArray = rects.toArray(new Rect[0]);

                MatOfRect boxes = new MatOfRect(boxesArray);

                MatOfInt indices = new MatOfInt();

                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
                // Draw result boxes:
                int[] ind = indices.toArray();
                int g = 0;
                List<Integer>  closePair1X = new ArrayList<>();
                List<Integer>  closePair1Y = new ArrayList<>();
                List<Integer>  s_closePair1X = new ArrayList<>();
                List<Integer>  s_closePair1Y = new ArrayList<>();
                List<Integer>  closePair2X = new ArrayList<>();
                List<Integer>  closePair2Y = new ArrayList<>();
                List<Integer>  s_closePair2X = new ArrayList<>();
                List<Integer>  s_closePair2Y = new ArrayList<>();


                for (int i = 0; i < ind.length; ++i) {
                    int idx = ind[i];
                    //int idGuy = clsIds.get(idx);
                    //Rect box = boxesArray[idx];
                    int cx1 = CentrosX.get(idx);
                    int cy1 = CentrosY.get(idx);
                    int w1 = Width_Arr.get(idx);
                    int h1 = Height_Arr.get(idx);
                    float conf = confs.get(idx);
                    int intConf = (int) (conf * 100);
                    //Imgproc.putText(frame,cocoNames.get(idGuy) + " " + intConf + "%",box.tl(),Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255,255,0),2);
                    //Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
                    Imgproc.circle(frame,new Point(cx1,cy1),5, new Scalar(255,0,0),-1);

                    for (int j = 0; j< ind.length;j++){
                        int idj = ind[j];
                        int cx2 = CentrosX.get(idj);
                        int cy2 = CentrosY.get(idj);
                        int w2 = Width_Arr.get(idj);
                        int h2 = Height_Arr.get(idj);
                        g = isClose(w1,h1,cx1,cy1,w2,h2,cx2,cy2);
                        if(g == 1){
                            closePair1X.add((int)cx1);
                            closePair1Y.add((int)cy1);
                            closePair2X.add((int)cx2);
                            closePair2Y.add((int)cy2);
                            status.set(idx,1);
                            status.set(idj,1);
                        }
                        else if(g == 2){
                            s_closePair1X.add((int)cx1);
                            s_closePair1Y.add((int)cy1);
                            s_closePair2X.add((int)cx2);
                            s_closePair2Y.add((int)cy2);
                            if (status.get(idx) != 1){
                                status.set(idx,1);
                            }
                            if (status.get(idj) != 1){
                                status.set(idj,1);
                            }
                        }
                    }
                }
                int total_p = CentrosX.size();
                int low_risk_p = Collections.frequency(status,2);
                int high_risk_p = Collections.frequency(status,1);
                int safe_p = Collections.frequency(status,0);
                int kk = 0;
                for (int i = 0; i < ind.length; ++i) {
                    int idx = ind[i];
/*
                    Imgproc.line(FR,new Point(0,H+10),new Point(FW,H+1),new Scalar(0,0,0),2);

                    Imgproc.putText(FR,"Social Distancing Analyser wrt. COVID-19",new Point(210,H+60),
                                    Core.FONT_HERSHEY_SIMPLEX,1,new Scalar(0,0,0),2  );

                    Imgproc.rectangle(FR, new Point(20,H+80), new Point(510,H+180), new Scalar(100, 100, 0), 2);

                    Imgproc.putText(FR,"Connecting lines shows closeness among people. ",new Point(30,H+100),
                                    Core.FONT_HERSHEY_SIMPLEX,0.6,new Scalar(100,100,0),2 );

                    Imgproc.putText(FR,"-- YELLOW: CLOSE",new Point(50,H+130),
                                    Core.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(0,170,170),2 );

                    Imgproc.putText(FR,"-- RED: VERY CLOSE",new Point(50,H+150),
                                    Core.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(0,0,250),2 );

                    Imgproc.rectangle(FR,new Point(535,H+80), new Point(1060,H+180), new Scalar(100, 100, 100), 2);

                    Imgproc.putText(FR,"Bounding box shows the level of risk to the person.",new Point(545,H+100),
                                    Core.FONT_HERSHEY_SIMPLEX,0.6,new Scalar(100,100,0),2 );

                    Imgproc.putText(FR,"-- DARK RED: HIGH RISK",new Point(565,H+130),
                                    Core.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(0,0,150),2 );

                    Imgproc.putText(FR,"-- ORANGE: LOW RISK",new Point(565,H+150),
                                    Core.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(0,120,255),2 );

                    Imgproc.putText(FR,"-- GREEN: SAFE",new Point(565,H+170),
                                    Core.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(0,150,0),2 );

                    String tot_str = "TOTAL COUNT: " + Integer.toString(total_p);
                    String high_str = "HIGH RISK COUNT: " + Integer.toString(high_risk_p);
                    String low_str = "LOW RISK COUNT: " + Integer.toString(low_risk_p);
                    String safe_str = "SAFE COUNT: " + Integer.toString(safe_p);

                    Imgproc.putText(FR,tot_str,new Point(10,H+25),
                                    Core.FONT_HERSHEY_SIMPLEX,0.6,new Scalar(0,0,0),2 );

                    Imgproc.putText(FR,safe_str,new Point(200,H+25),
                                    Core.FONT_HERSHEY_SIMPLEX,0.6,new Scalar(0,170,0),2 );

                    Imgproc.putText(FR,low_str,new Point(380,H+25),
                                    Core.FONT_HERSHEY_SIMPLEX,0.6,new Scalar(0,120,255),2 );

                    Imgproc.putText(FR,high_str,new Point(630,H+25),
                                    Core.FONT_HERSHEY_SIMPLEX,0.6,new Scalar(0,0,150),2 );
*/
                    Rect box = boxesArray[idx];
                    if (status.get(kk) == 1){
                        Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(150, 0, 0), 2);
                    }
                    else if (status.get(kk) == 0){
                        Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 255, 0), 2);
                    }
                    else{
                        Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 120, 0), 2);
                    }
                    kk++;
                }
                for (int h = 0; h< closePair1X.size();h++){
                    Imgproc.line(frame,new Point(closePair1X.get(h),closePair1Y.get(h)),new Point(closePair2X.get(h),closePair2Y.get(h)),new Scalar(0,0,255),2);
                }
                for (int b = 0; b< s_closePair1X.size();b++){
                    Imgproc.line(frame,new Point(s_closePair1X.get(b),s_closePair1Y.get(b)),new Point(s_closePair2X.get(b),s_closePair2Y.get(b)),new Scalar(0,255,255),2);
                }

                //frame.copyTo(FR);
                //return FR;
            }

        }



        return frame;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        if (startYolo == true){

            String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg" ;
            String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";

            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);


        }



    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }

    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    static int isClose(int w1,int h1, int cx1, int cy1, int w2, int h2, int cx2, int cy2){
        double c_d = dist(cx1,cy1,cx2,cy2);
        int aw = 0;
        int ah = 0;
        if(h1<h2){
            aw = w1;
            ah = h1;
        }
        else {
            aw = w2;
            ah = h2;
        }

        double T = ((double)cy2 - (double)cy1) / ((double)cx2 - (double)cx1);
        if (Double.isInfinite(T)){
            T = 1.633123935319537E16;
        }
        double S = T2S(T);
        double C = T2C(T);
        double d_hor = C*c_d;
        double d_ver = S*c_d;
        double vc_calib_hor = aw;
        double vc_calib_ver = ah*0.4*0.8;
        double c_calib_hor = aw *1.7;
        double c_calib_ver = ah*0.2*0.8;

        if(((0<d_hor)&&(d_hor<vc_calib_hor)) && ((0<d_ver)&&(d_ver<vc_calib_ver))){
            return 1;
        }
        else if(((0<d_hor)&&(d_hor<c_calib_hor)) && ((0<d_ver)&&(d_ver<c_calib_ver))){
            return 2;
        }
        else {
            return 0;
        }

    }

    static double dist(int px1, int py1, int px2, int py2){
        return Math.pow((Math.pow(px1-px2,2)+Math.pow(py1-py2,2)),0.5);
    }
    static double T2S(double T){

        return Math.abs((T/Math.pow((1+Math.pow(T,2)),0.5)));
    }

    static double T2C(double T){
        return Math.abs((1/Math.pow((1+Math.pow(T,2)),0.5)));
    }
}
