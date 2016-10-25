#include<bits/stdc++.h>

using namespace std;

struct triple{
	string subject;
	string property;
	string object;	
	triple(string sub, string pro, string obj): subject(sub),property(pro),object(obj) {
	}
	triple(const triple &t){
		subject = t.subject;
		property = t.property;
		object = t.object;
	}
	void rewrite(){
	}	
	string toString(){
		return subject + " " + property +  " " + object;
	}
};

struct pattern{
	string subject;
	string property;
	string object;		
	bool inverted;
	bool transS;
	bool transO;
	pattern(string sub, string pro, string obj): subject(sub),property(pro),object(obj) {
		inverted = false;
		transS = false;
		transO = false;
	}
	pattern(const triple &t){
		subject = t.subject;
		property = t.property;
		object = t.object;
	}	
	string toString(){
		return subject + " " + property +  " " + object;
	}
};

typedef vector<triple> rule;

vector<rule> ruleSet;

triple getRightHand(rule r){
	triple result( r[r.size()-1] );
	return result;
}

vector<triple> getLeftHand(rule r){
	vector<triple> result;
	copy( r.begin(), r.end()-1, back_inserter( result ) );
	return result;
}


bool isMatched(pattern target, triple obj){ //Only write for property, extending easily for subject & object
	if( obj.property.length() > 2 && obj.property!= target.property ) return false;
	return true;
}

bool isMatched(pattern target, pattern obj){ //Only write for property, extending easily for subject & object
	if( obj.property.length() > 2 && obj.property!= target.property ) return false;
	return true;
}

void readRuleSet(){
	string ruleDes;	
	string subject,property,object,connector;	
	while( getline(cin,ruleDes) ){
		istringstream is(ruleDes);
		rule r; 
		while(is >> subject ){		
			is >> property;		
			is >> object;			
			triple t(subject,property,object);			
			r.push_back(t);			
		}		
		ruleSet.push_back(r);		
	}	
}

vector<pattern> extendTripleFromRule(string query){
	istringstream is(query);
	string subject,property,object;
	is >> subject; is >> property; is >> object;
	pattern target(subject,property,object);
	vector<pattern> result;
	queue<pattern> q;
	
	q.push(target);
	
	while( !q.empty() ){
		pattern current = q.front();
		//cout << "current = " <<  current.subject <<  " " << current.property << " " << current.object << endl;
		q.pop();
		bool appearance_flag = false;
		for(int i=0; i<result.size(); i++){
			if( isMatched(current, result[i]) ){
				appearance_flag=true;
				break;
			}
		}
		if(appearance_flag) continue;
		
		result.push_back(current);
		
		
		for(int i=0; i<ruleSet.size(); i++){
			triple right = getRightHand(ruleSet[i]);
			if( isMatched(current,right) ){
				// Generate dictionary
				map<string,string> mp;
				mp[right.subject] = current.subject;
				mp[right.object] = current.object;
				
				// Get leftthand size
				vector<triple> left = getLeftHand(ruleSet[i]);
				// Update RightHand Size 				
				if( left.size()==1 ){					
					string subject = left[0].subject, object = left[0].object, property = left[0].property;
					if( mp[subject].length() ) subject = mp[subject];
					if( mp[object].length() ) object = mp[object];
					pattern res(subject,property,object);
					if( isMatched(current,left[0]) ) res.inverted = true;
					q.push(res);
				}else{
					for(int i=0; i<left.size(); i++){
						string subject = left[i].subject, object = left[i].object, property = left[i].property;
						cout << subject << " " << property << " " << object << endl;
						if( mp[subject].length() ) subject = mp[subject];
						if( mp[object].length() ) object = mp[object];
						pattern res(subject,property,object);
						if( subject == current.subject ) res.transO = true;
						if( object == current.object ) res.transS = true;
						q.push(res);
					}
				}				
			}
		}		
	}
	return result;	
}

int main(){
	freopen("rule.txt","r",stdin);
	readRuleSet();
	for(int i=0; i<ruleSet.size(); i++){
		rule r = ruleSet[i];
		cout << r[0].toString();
		for(int i=1; i<r.size()-1; i++ ){
			cout << " AND " << r[i].toString();
		}
		cout << " THEN " << r[r.size()-1].toString() << endl;		
	}
	vector<pattern> vt =  extendTripleFromRule("a rdfs:subPropertyOf x:Tome");
	for(int i=0; i<vt.size(); i++){
		cout << vt[i].subject << " " << vt[i].property << " " << vt[i].object << " " << vt[i].inverted << " " << vt[i].transS << " " << vt[i].transO << endl;
	}
	return 0;
}
