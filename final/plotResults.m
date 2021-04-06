function plotResults(dim_poly,Xout,X_SOS,Y_SOS,X_DeepSDP,Y_DeepSDP)
    
%h_rec2 = plot([-5,-4.99],[0,0],'m','LineWidth',2);hold on; %plot([-5,5], [X_max,X_max],'m','LineWidth',3)
temp = plot([-5,-4.99],[0,0],'b','LineWidth',2);hold on; %plot([-5,5], [X_max,X_max],'m','LineWidth',3)
data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.','MarkerEdgeColor','b'); hold on;

for i = 1:dim_poly-1
    h_SOS1 = plot([X_SOS(i),X_SOS(i+1)],[Y_SOS(i),Y_SOS(i+1)],'red','LineWidth',4);
end
h_SOS2 = plot([X_SOS(dim_poly),X_SOS(1)],[Y_SOS(dim_poly),Y_SOS(1)],'red','LineWidth',4,'DisplayName','NNPsatz');

hold on;

for i = 1:dim_poly-1
    h_DeepSDP1 = plot([X_DeepSDP(i),X_DeepSDP(i+1)],[Y_DeepSDP(i),Y_DeepSDP(i+1)],'black','LineWidth',4);hold on;
end
h_DeepSDP2 = plot([X_DeepSDP(dim_poly),X_DeepSDP(1)],[Y_DeepSDP(dim_poly),Y_DeepSDP(1)],'black','LineWidth',4,'DisplayName','DeepSDP');

% Plot true values
%plot([IBP_max(1),IBP_max(1)],[IBP_max(1),IBP_max(1)])
%h_rec = rectangle('Position',[IBP_min(1), IBP_min(2), IBP_max(1) - IBP_min(1), IBP_max(2) - IBP_min(2)],'EdgeColor','m','LineWidth',4)

%legend([temp,h_SOS2,h_DeepSDP2,h_rec2],{'True Values','NNPsatz','DeepSDP','IBP'},'FontSize',20)
legend([temp,h_SOS2,h_DeepSDP2],{'True Values','NNPsatz','DeepSDP'},'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'));
ax2 = get(gca,'XTickLabel');
set(gca,'XTickLabel',ax2,'fontsize',22)

%legend('True Output','NNPsatz','DeepSDP','IBP')

end
