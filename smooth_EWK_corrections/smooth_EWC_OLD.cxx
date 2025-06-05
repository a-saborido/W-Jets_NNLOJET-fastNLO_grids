#include <cmath>
#include <algorithm>

// from
// https://chatgpt.com/share/6832d647-7910-8003-a795-fe1f9cb642b7
TH1D* SmoothHistogramWithErrors(const TH1D* h) {
    int nBins = h->GetNbinsX();
    TH1D* h_smoothed = (TH1D*)h->Clone("h_smoothed");
    h_smoothed->Reset();

    for (int i = 1; i <= nBins; ++i) {
        double sumWeightedContent = 0;
        double sumWeights = 0;
        double sumErrorSquared = 0;

        for (int j = i - 2; j <= i + 2; ++j) {
            if (j < 1 || j > nBins) continue;

            double content = h->GetBinContent(j);
            double error   = h->GetBinError(j);
            double width   = h->GetBinWidth(j);

            if (error <= 0 || width <= 0) continue;

            double weight = 1.0 / (error * error * width);

            sumWeightedContent += weight * content;
            sumWeights         += weight;
            sumErrorSquared    += weight * weight * error * error;
        }

        if (sumWeights > 0) {
            double smoothedContent = sumWeightedContent / sumWeights;
            double smoothedError   = std::sqrt(sumErrorSquared) / sumWeights;

            h_smoothed->SetBinContent(i, smoothedContent);
            h_smoothed->SetBinError(i, smoothedError);
        }
    }

    return h_smoothed;
}



void print_hist(const TH1D* hist) {
   cout<<endl;
   printf("  %8s  %8s  %8s  %8s\n","bin_lo","bin_up","value","error[%]");
   for ( int i = 1 ; i<=hist->GetNbinsX() ; i++) {
      printf("  %8.4g  %8.4g  %8.4f  %8.2f\n",
	     hist->GetBinLowEdge(i),
	     hist->GetBinLowEdge(i+1),
	     hist->GetBinContent(i),
	     hist->GetBinError(i)*100. );
   }

}


void smooth_EWC_OLD() {

   // one-line print
   TFile::Open("EWC.root")->Get<TH1D>("EWC/jet_pt1_1j")->Print("all");

   cout<<endl;
   
   // step-by-step and print nicer
   TFile* _file0 = TFile::Open("EWC.root");
   _file0->cd("EWC");
   TH1D* hist = gDirectory->Get<TH1D>("HT_1j_Wp");
   print_hist(hist);


   // call a manual smoothing algo
   TH1D* smoothedhist = SmoothHistogramWithErrors(hist);
   
   // smooth and print nicer
   // NB: smooth changes the original histogram in memory
   hist->Smooth(2);
   print_hist(hist);

   print_hist(smoothedhist);
   
   return;
   
}
